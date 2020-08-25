__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from math import pi, cos, sin, floor, ceil, sqrt

# External packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as apyfits
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.modeling.models import Gaussian2D
from astropy.convolution import convolve
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from ccdproc import trim_image, wcs_project
from photutils import SplitCosineBellWindow, CosineBellWindow, \
    data_properties, centroid_com, create_matching_kernel
import skimage.transform as skt

# From this package:
from mp_phot import util

PINPOINT_PLTSOLVD_TEXT = 'Plate has been solved by PinPoint'
PIXEL_FACTOR_HISTORY_TEXT = 'Pixel scale factor applied by apply_pixel_scale_factor().'
RADIANS_PER_DEGREE = pi / 180.0
MAX_MISALIGNMENT_FOR_CONVERGENCE = 1000.0  # in millipixels

def measure_mp():
    pass


def measure_comps():
    pass


CLASS_DEFINITIONS__________________________________________ = 0


class MP_Image:
    """ An astropy CCDData object, embellished with image-associated data we will need."""
    def __init__(self, directory, filename, settings):
        """
        :param directory: path to directory holding FITS files. [string]
        :param filename: FITS filename. [string]
        """
        self.fullpath = os.path.join(directory, filename)
        self.filename = filename
        hdu0 = apyfits.open(self.fullpath)[0]
        h = hdu0.header

        # Replace obsolete key if present.
        radesysa = h.get('RADECSYS')
        if radesysa is not None:
            h['RADESYSa'] = radesysa
            del h['RADECSYS']

        # Apply corrective pixel scale factor if solved by PinPoint:
        plt_solved_value = h.get('PLTSOLVD', False)
        pltsolvd_is_strictly_true = plt_solved_value and isinstance(plt_solved_value, bool)
        if pltsolvd_is_strictly_true:
            is_solved_by_pinpoint = (h.comments['PLTSOLVD'] == PINPOINT_PLTSOLVD_TEXT)
        else:
            is_solved_by_pinpoint = False
        pixel_scale_factor_is_applied = any([val == PIXEL_FACTOR_HISTORY_TEXT for val in h['HISTORY']])
        if is_solved_by_pinpoint and not pixel_scale_factor_is_applied:
            h = apply_pixel_scale_factor(h, settings.get('PINPOINT_PIXEL_SCALE_FACTOR', 1.00))
            h['HISTORY'] = PIXEL_FACTOR_HISTORY_TEXT

        # Construct CCDData object:
        this_wcs = WCS(header=h)
        self.image = CCDData(data=hdu0.data, wcs=this_wcs, header=h, unit='adu')
        self.filter = self.image.meta['Filter']
        self.exposure = self.image.meta['Exposure']
        self.jd = self.image.meta['JD']
        self.jd_mid = self.jd + (self.exposure / 24 / 3600) / 2
        self.ref_star_radecs = []
        self.mp_radec = None

    def __str__(self):
        return 'MP_Image object from ' + self.filename


class MP_ImageList:
    """ A list of MP images, embellished with data common to all the images.
        Contains all FITS information for one session (one an "Astronight", one minor planet "MP").
        Each image is held in a MP_Image object.
        Images are ordered by increasing JD (normal time order). """
    def __init__(self, directory, mp_id, an, filter, images, ref_star_locations, mp_locations, settings):
        """ Nominal constructor, NOT usually invoked directly;
            Normally use MP_ImageList.from_fits() or mp_imagelist.render_subimagelist() instead.
            :param directory: directory where MP FITS files were originally found. [string]
            :param mp_id:
            :param an:
            :param filter:
            :param images:
            :param ref_star_locations: as from do_workflow.Control(). [list of tuples]
            :param mp_locations: as from do_workflow.Control(). [list of tuples]
            :param settings: global settings from calling code. [.do_workflow.Settings object]
            """
        self.directory = directory
        self.mp_id = str(mp_id)
        self.mp_label = 'MP_' + str(mp_id)
        self.an = an
        self.filter = filter
        # Ensure images are limited to those in filter specified:
        filter_images = [im for im in images if im.filter == filter]
        # Ensure images are sorted to chronological:
        jds = [im.jd_mid for im in filter_images]
        as_tuples = [(im, jd) for (im, jd) in zip(filter_images, jds)]
        as_tuples.sort(key=lambda tup: tup[1])  # in-place sort by increasing JD.

        self.mp_images = list(zip(*as_tuples))[0]
        self.filenames = [im.filename for im in self.mp_images]
        self.jd_mids = [im.jd_mid for im in self.mp_images]
        self.ref_star_locations = ref_star_locations
        self.ref_star_radecs = []                  # "
        self.mp_locations = mp_locations
        self.mp_radecs = []              # placeholder
        self.mp_locations_all = []       # "
        self.settings = settings

        self.subimages = []  # placeholder
        self.subimage_ref_star_locations = []  # will be a list of lists of (x,y) tuples.
        self.subimage_mp_locations = []

        # Set up (private) dict of images by filename key:
        self._dict_of_mp_images = dict()
        for mp_image in self.mp_images:
            self._dict_of_mp_images[mp_image.filename] = mp_image

    def __str__(self):
        return 'MP_ImageList object of ' + self.mp_label + ' with ' + str(len(self.mp_images)) + ' images.'

    @classmethod
    def from_fits(cls, directory, mp_id, an, filter, ref_star_locations, mp_locations, settings):
        """ Alternate constructor--probably the most used.
            Contains all FITS information for one session (one an "Astronight", one minor planet "MP").
            Each image is held in a MP_Image object, which is a CCDData image with some additional data.
            Images are ordered by increasing JD (normal time order).
        :param directory: directory where MP FITS files are found. [string]
        :param mp_id:
        :param an:
        :param filter:
        :param ref_star_locations: as read from control.txt by do_workflow.Control(). [list of tuples]
        :param mp_locations: as read from control.txt by do_workflow.Control(). [list of tuples]
        :param settings: global settings from calling code. [.do_workflow.Settings object]

        """
        all_filenames = util.get_mp_filenames(directory)  # order not guaranteed.
        all_images = [MP_Image(directory, fn, settings) for fn in all_filenames]
        # Sort filenames and images to chronological:
        all_jds = [im.jd_mid for im in all_images]
        as_tuples = [(im, jd) for (im, jd) in zip(all_images, all_jds)]
        as_tuples.sort(key=lambda tup: tup[1])  # in-place sort by increasing JD.
        as_list = list(zip(*as_tuples))
        sorted_images = as_list[0]
        images_in_filter = [im for im in sorted_images if im.filter == filter]
        return cls(directory, mp_id, an, filter, images_in_filter,
                   ref_star_locations, mp_locations, settings)

    # Allow direct access as image = imlist['some_filename'].
    def __getitem__(self, filename):
        return self._dict_of_mp_images.get(filename, None)  # return None if key (filename) is absent.

    def calc_ref_star_radecs(self):
        """ Use WCS (plate solution) to find ref stars' RA,Dec, given centers (x,y) from one image. """
        radius = 4 * self.settings['FWHM_NOMINAL']  # 1/2 size of array including background estimation.
        mask_radius = radius / 2                    # radius of area for flux centroid estimation.
        for (filename, x0, y0) in self.ref_star_locations:
            mp_image = self[filename]
            square = util.Square(mp_image.image, x0, y0, radius, mask_radius)
            xc, yc = square.recentroid()
            ra, dec = tuple(mp_image.image.wcs.all_pix2world([list((xc, yc))], 0)[0])
            self.ref_star_radecs.append((ra, dec))

    def calc_mp_radecs(self):
        """ Use WCS (plate solution) to find MP's RA,Dec in *all* images, given MP center (x,y) in 2. """
        radius = 8 * self.settings['FWHM_NOMINAL']  # 1/2 size of array including background estimation.
        mask_radius = radius / 2                    # radius of area for flux centroid estimation.
        jd_mids, ra_values, dec_values = [], [], []
        for (filename, x0, y0) in self.mp_locations[:2]:
            this_image = self[filename].image
            sq = util.Square(this_image, x0, y0, radius, mask_radius)
            xc, yc = sq.recentroid()  # TODO: is recentroiding MP wise? I doubt it. ********************
            ra, dec = tuple(this_image.wcs.all_pix2world([list((xc, yc))], 0)[0])
            jd_mids.append(self[filename].jd_mid)
            ra_values.append(ra)
            dec_values.append(dec)
        span_seconds = (jd_mids[1] - jd_mids[0]) * 24 * 3600
        ra_per_second = (ra_values[1] - ra_values[0]) / span_seconds
        dec_per_second = (dec_values[1] - dec_values[0]) / span_seconds
        for mp_image in self.mp_images:
            dt = (mp_image.jd_mid - jd_mids[0]) * 24 * 3600
            ra = ra_values[0] + dt * ra_per_second
            dec = dec_values[0] + dt * dec_per_second
            x, y = tuple(mp_image.image.wcs.all_world2pix([[ra, dec]], 0, ra_dec_order=True)[0])
            self.mp_radecs.append((ra, dec))
            self.mp_locations_all.append((x, y))

    def make_subimages(self):
        """ Aligns only by whole-pixel (integer) shifts in x and y, no rotation or interpolation.
            Very approximate, for initial alignment (and then cropping) use only."""
        radius = 8 * self.settings['FWHM_NOMINAL']  # 1/2 size of array including background estimation.
        mask_radius = radius / 2                    # radius of area for flux centroid estimation.

        # Determine offsets, (recentroided and averaged) from ref_star_positions:
        x_means, y_means = [], []
        x_ref_star_raw, y_ref_star_raw = [], []
        for mi in self.mp_images:
            xcs, ycs = [], []
            for i, radec in enumerate(self.ref_star_radecs):
                x0, y0 = tuple(mi.image.wcs.all_world2pix([radec], 0, ra_dec_order=True)[0])
                sq = util.Square(mi.image, x0, y0, radius, mask_radius)
                xc, yc = sq.recentroid()
                xcs.append(xc)
                ycs.append(yc)
            x_means.append(np.mean(xcs))
            y_means.append(np.mean(ycs))
            x_ref_star_raw.append(xcs)  # for bounding box calcs
            y_ref_star_raw.append(ycs)  # "
        x_offsets = [int(round(xm - x_means[0])) for xm in x_means]  # these look good.
        y_offsets = [int(round(ym - y_means[0])) for ym in y_means]  # these look good.

        # Calculate x,y:min,max for each mp_image, save as rel to mp_images[0]:
        x_min, x_max, y_min, y_max = [], [], [], []
        for i, mi in enumerate(self.mp_images):
            x_values = x_ref_star_raw[i].copy()  # ref star positions were calculated above.
            y_values = x_ref_star_raw[i].copy()  # "
            x_values.append(self.mp_locations_all[i][0])
            y_values.append(self.mp_locations_all[i][1])
            x_min.append(min(x_values) - x_offsets[i])
            x_max.append(max(x_values) - x_offsets[i])
            y_min.append(min(y_values) - y_offsets[i])
            y_max.append(max(x_values) - y_offsets[i])

        # Calculate global x,y:min,max (rel to mp_images[0]), then recast relative to each mp_image:
        x_min_global_0 = int(floor(min(x_min)))
        x_max_global_0 = int(ceil(max(x_max)))
        y_min_global_0 = int(floor(min(y_min)))
        y_max_global_0 = int(ceil(max(y_max)))
        subimage_margin = int(ceil(2 + 12 * self.settings['FWHM_NOMINAL']))
        x_min_bounds = [x_min_global_0 - subimage_margin + x_offset for x_offset in x_offsets]
        x_max_bounds = [x_max_global_0 + subimage_margin + x_offset for x_offset in x_offsets]
        y_min_bounds = [y_min_global_0 - subimage_margin + y_offset for y_offset in y_offsets]
        y_max_bounds = [y_max_global_0 + subimage_margin + y_offset for y_offset in y_offsets]

        # Make subimages [new CCDData objects], save in MP_ImageList 'self.subimages':
        # Use native CCDData slicing (which keeps WCS and makes copy), not shift_2d_array().
        for i, mi in enumerate(self.mp_images):
            # Make subimage CCDData object from MP_Image's internal CCDData object:
            subimage = trim_image(mi.image[y_min_bounds[i]:y_max_bounds[i] + 1,
                                           x_min_bounds[i]:x_max_bounds[i] + 1])
            self.subimages.append(subimage)

        # For TEST only:
        # plot_subimages('TEST new subimages', self)

    def wcs_align_subimages(self):
        """ Use WCS to align to one image all other images, by interpolation. Uses ccdproc.wcs_project().
            May include rotation. Tighter than make_subimages() above which uses whole-pixel slicing,
                but not so tight as ref_star_align_all() below which uses convolution. """
        wcs_reference = self.subimages[0].wcs.copy()
        aligned_subimages = [wcs_project(si, wcs_reference, order='biquadratic') for si in self.subimages]
        # aligned_subimages = [wcs_project(si, wcs_reference) for si in self.subimages]
        self.subimages = aligned_subimages

    def trim_nans_from_subimages(self):
        """ Find coordinates of (largest) bounding box within which all images have all good pixels,
            meaning no NaN pixels at the data edges, no True (masked) pixels in the mask edges.
            Works from the outside edges and inward, so will probably miss *internal* NaN
                or masked pixels.
            Adapted 2020-08-24 from mpc.mp_bulldozer.recrop_mp_images().
            """
        # All images must stay aligned and of same size and of correct WCS,
        #    so they must be tested and trimmed *together* using ccdproc.trim_image().
        arrays = [si.data for si in self.subimages]
        if not all([a.shape == arrays[0].shape for a in arrays]):
            print(' >>>>> ERROR .trim_nans_from_subimages(): the ' + str(len(arrays)) +
                  ' arrays differ in shape, but must be uniform shape.')
            return None
        pixel_sum = np.sum(a for a in arrays)  # for each pixel: if any array is NaN, then sum will be NaN.
        # is_any_nan = np.full_like(arrays[0], False, np.bool)  # accumulator.
        # for array in arrays:
        #     is_any_nan = np.logical_or(is_any_nan, np.isnan(array))

        # Find largest rectangle having no NaN in any of the arrays, inward spiral search:
        x_min, y_min = 0, 0
        y_max, x_max = arrays[0].shape
        while min(x_max - x_min, y_max - y_min) > 20 * self.settings['FWHM_NOMINAL']:
            before_limits = (x_min, x_max, y_min, y_max)
            if np.any(np.isnan(pixel_sum[y_min:y_min + 1, x_min:x_max])):  # if top row has NaN.
                y_min += 1
            if np.any(np.isnan(pixel_sum[y_max - 1:y_max, x_min:x_max])):  # if bottom row has NaN.
                y_max -= 1
            if np.any(np.isnan(pixel_sum[y_min:y_max, x_min:x_min + 1])):  # if leftmost column has NaN.
                x_min += 1
            if np.any(np.isnan(pixel_sum[y_min:y_max, x_max - 1:x_max])):  # if rightmost col has NaN.
                x_max -= 1
            after_limits = (x_min, x_max, y_min, y_max)
            if after_limits == before_limits:  # if no trims made in this iteration.
                break
        if np.any(np.isnan(pixel_sum[y_min:y_max, x_min:x_max])):
            print(' >>>>> ERROR: recrop_mid_images() did not succeed in removing all NaNs.')
        nan_trimmed_subimages = [trim_image(si[y_min:y_max, x_min:x_max]) for si in self.subimages]
        self.subimages = nan_trimmed_subimages

    def get_subimage_locations(self):
        """ Get exact ref_star x,y positions from radecs by WCS, then recentroiding.
            Get best MP x,y positions by WCS, do not recentroid. Probably test middle positions against
            those expected by time interpolation.
        """
        # Get all subimage ref star locations:
        radius = 8 * self.settings['FWHM_NOMINAL']  # 1/2 size of array including background estimation.
        mask_radius = radius / 2                    # radius of area for flux centroid estimation.
        for si in self.subimages:
            this_subimage_ref_star_locations = []
            for radec in self.ref_star_radecs:
                x0, y0 = tuple(si.wcs.all_world2pix([radec], 0, ra_dec_order=True)[0])
                sq = util.Square(si, x0, y0, radius, mask_radius)
                xc, yc = sq.recentroid()
                this_subimage_ref_star_locations.append((xc, yc))
            self.subimage_ref_star_locations.append(this_subimage_ref_star_locations)

        # Get all subimage MP locations:
        for i, si in enumerate(self.subimages):
            radec = self.mp_radecs[i]
            x, y = tuple(si.wcs.all_world2pix([radec], 0, ra_dec_order=True)[0])
            self.subimage_mp_locations.append((x, y))
        # TEST ONLY: test subimage MP locations (ex earliest and latest) for time-interpolation accuracy:
        xy_earliest = self.subimage_mp_locations[0]
        xy_latest = self.subimage_mp_locations[-1]
        for i, mploc in enumerate(self.subimage_mp_locations[1:-1]):
            jd_fraction = (self.jd_mids[i] - self.jd_mids[0]) / (self.jd_mids[-1] - self.jd_mids[0])
            x_expected = xy_earliest[0] + jd_fraction * (xy_latest[0] - xy_earliest[0])
            y_expected = xy_earliest[1] + jd_fraction * (xy_latest[1] - xy_earliest[1])
            x_actual = self.subimage_mp_locations[i][0]
            y_actual = self.subimage_mp_locations[i][1]
            distance = sqrt((x_actual - x_expected)**2 + (y_actual - y_expected)**2)
            print('{:3d}'.format(i) + ':',
                  '  expected=', '{:8.3f}'.format(x_expected), '{:8.3f}'.format(x_expected),
                  '  actual=', '{:8.3f}'.format(x_actual), '{:8.3f}'.format(y_actual),
                  '  dist=', '{:8.3f}'.format(distance), 'pixels.')

    def make_subarrays(self):
        subarrays = []
        for i, si in enumerate(self.subimages):
            subarray = Subarray(filename=self.filenames[i],
                                array=si.data,
                                mask=si.mask,
                                jd_mid=self.jd_mids[i],
                                exposure=self.jd_mids[i],
                                ref_star_locations=self.subimage_ref_star_locations[i],
                                mp_location=self.subimage_mp_locations[i])
            subarrays.append(subarray)
        subarray_list = SubarrayList(subarrays=subarrays,
                                     mp_id=self.mp_id,
                                     an=self.an,
                                     filter=self.filter,
                                     settings=self.settings)
        return subarray_list


class Subarray:
    def __init__(self, filename, array, mask, jd_mid, exposure, ref_star_locations, mp_location):
        """ Hold one subarray and matching mask, for an image segment.  For late-stage processing.
            NO WCS, no RA,Dec data for MP or ref stars.
        :param filename: [string]
        :param array: [ndarray of floats]
        :param mask: [ndarray of bools]
        :param jd_mid: [float]
        :param exposure: [float]
        :param ref_star_locations: [list of floats]
        :param mp_location: [float]
        """
        self.filename = filename
        self.array = array.copy()
        if mask is None:
            self.mask = np.full_like(self.array, False, np.bool)
        else:
            self.mask = mask.copy()
        self.jd_mid = jd_mid
        self.exposure = exposure
        self.ref_star_locations = ref_star_locations.copy()
        self.mp_location = mp_location
        self.ref_star_squares = []   # placeholder, will be a list of util.Square objects.
        self.matching_kernel = None  # placeholder, will be a np.ndarray
        self.convolved_array = None  # placeholder, will be a np.ndarray
        self.realigned_array = None  # placeholder, will be a np.ndarray

    def __str__(self):
        return 'Subarray object from ' + self.filename + ' of shape ' + str(self.array.shape)

    def align_on_ref_stars(self):
        pass  # TODO: use scikit-image.transform functions to find and apply similary transform.


class SubarrayList:
    def __init__(self, subarrays, mp_id, an, filter, settings):
        self.subarrays = subarrays
        self.mp_id = mp_id
        self.an = an
        self.filter = filter
        self.settings = settings

        # Set up (private) dict of subarrays by filename key:
        self._dict_of_subarrays = dict()
        for sa in self.subarrays:
            self._dict_of_subarrays[sa.filename] = sa

    def __str__(self):
        return 'SubarrayList of ' + str(len(self.subarrays)) + ' subarrays.'

    # Allow direct access as subarray = subarraylist['some_filename'].
    def __getitem__(self, filename):
        return self._dict_of_subarrays.get(filename, None)  # return None if key (filename) is absent.

    def make_matching_kernels(self):
        """ Find the smallest appropriate target 2-D Gaussian PSF, then for each subarray,
            make a matching kernel that will transform the ref stars into that 2-D Gaussian.
            Adapted closely 2020-08-24 from mpc.mp_bulldozer.make_ref_star_psfs(),
            .calc_target_kernel_sigma(), and .calc_target_kernels().
        :return: None. Subarraylist.subarray objects are populated with matching kernels.
                 [list of lists of numpy.ndarrays]
        """
        # Make ref star PSF arrays (very small kernel-sized images) [save as util.Square objects]:
        radius = 5 * self.settings['FWHM_NOMINAL']
        for sa in self.subarrays:
            subarray_psfs = []  # for this subarray (all ref stars).
            for x_center, y_center in sa.ref_star_locations:
                square = util.Square(sa.array, x_center, y_center, radius)
                bkgd_adus, _ = util.calc_background_adus(square.data)
                bkdg_subtracted_array = square.data - bkgd_adus
                # Vignette bkgd-subtracted array (reduce noise far from center, in lieu of sharp mask).
                window_array = (SplitCosineBellWindow(alpha=0.5, beta=0.25))(square.shape)
                raw_kernel = window_array * bkdg_subtracted_array
                square.data = [raw / np.sum(raw) for raw in raw_kernel]  # normalized kernel.
                sa.ref_star_squares.append(square)
                subarray_psfs.append(square)
            sa.ref_star_squares.append()

        # Calculate smallest appropriate sigma for target 2_D Gaussian PSF (everything scales from this):
        sigma_list = []
        all_psfs = [sa.ref_star_squares for sa in self.subarrays]
        for arr in np.flatten(all_psfs):
            dps = data_properties(arr.data)
            sigma_list.append(dps.semimajor_axis_sigma.value)
            print('   ',  '{:.2f}'.format(dps.xcentroid.value),
                          '{:.2f}'.format(dps.ycentroid.value),
                          '{:.3f}'.format(dps.semimajor_axis_sigma.value))
        max_sigma = max(sigma_list)
        target_sigma = 1.0 * max_sigma

        # Make matching kernels:
        for i_sa, sa in enumerate(self.subarrays):
            sa_matching_kernels = []
            for i_rs, rs in enumerate(sa.ref_star_locations):
                edge_length = sa.shape[0]
                y, x = np.mgrid[0:edge_length, 0:edge_length]
                this_psf = all_psfs[i_sa][i_rs]
                x_center_parent, y_center_parent = this_psf.recentroid()  # relative to parent.
                x_center0 = x_center_parent - this_psf.x_low
                y_center0 = y_center_parent - this_psf.y_low
                # Make target PSFs (2-D Gaussian):
                gaussian = Gaussian2D(1, x_center0, y_center0, target_sigma, target_sigma)  # function
                target_psf = gaussian(x, y)  # ndarray
                target_psf /= np.sum(target_psf)  # ensure normalized to sum 1.
                matching_kernel = create_matching_kernel(this_psf, target_psf, CosineBellWindow(alpha=0.35))
                sa_matching_kernels.append(matching_kernel)
            sum_matching_kernels = np.sum(mk for mk in sa_matching_kernels)
            sa.matching_kernel = sa_matching_kernels / np.sum(sum_matching_kernels)  # normlized to sum 1.

    def convolve_subarrays(self):
        """ Convolve subarrays with matching kernels to render new subarrays with very nearly Gaussian
            ref star PSFs with very nearly uniform sigma.
        :return: None. Subarraylist.subarray objects are populated with convolved arrays.
                 [list of lists of numpy.ndarrays]
        """
        new_subarrays = []
        for i_sa, sa in enumerate(self.subarrays):
            convolved_array = convolve(sa.array.copy(), sa.matching_kernel, boundary='extend')
            sa.convolved_array = Subarray(sa.filename, convolved_array, None, sa.jd_mid,
                                          sa.exposure, sa.ref_star_locations, sa.mp_locations)

    def realign(self, max_iterations=3):
        """ Iteratively improve array alignments to millipixel precision.
            Each loop: (1) get best similarity transform (rotation+translation+scale)
            from ref star positions vs desired positions, (2) perform transform.
        :param max_iterations: [int]
        :return: None.
        """
        # Set target ref star pixel x,y locations as the mean recentroided location, across subarrays:
        radius = 5 * self.settings['FWHM_NOMINAL']
        target_locations = []
        for i_rs in range(len(self.subarrays[0].ref_star_locations)):
            x_list, y_list = [], []
            for sa in self.subarrays:
                square = util.Square(sa.realigned_array,
                                     sa.ref_star_locations[i_rs][0],
                                     sa.ref_star_locations[i_rs][1], radius)
                x, y = square.recentroid()
                x_list.append(x)
                y_list.append(y)
            target_locations.append(tuple([np.mean(x_list), np.mean(y_list)]))
        target_locations = np.array(target_locations)

        # Initialize the best/current subarrays
        current_arrays = [sa.realigned_array for sa in self.subarrays]

        for i_loop in range(max_iterations):
            # Estimate misalignment (in millipixels RMS):
            total_squared_misalignment = 0.0
            n_centers = 0
            current_locations = []  # will be list[subarray][ref_star](x,y tuple).
            for sa in self.subarrays:
                current_location = []
                for i_rs in range(len(sa.ref_star_locations)):
                    square = util.Square(sa.realigned_array,
                                         sa.ref_star_locations[i_rs][0],
                                         sa.ref_star_locations[i_rs][1], radius)
                    x, y = square.recentroid()
                    current_location.append((x, y))
                    x_target, y_target = target_locations[i_rs]
                    total_squared_misalignment += 1000.0 * ((x - x_target)**2 + (y - y_target)**2)
                    n_centers += 1
                current_locations.append(current_location)
            rms_misalignment = sqrt(total_squared_misalignment / n_centers)

            # TODO: Print i_loop, old xy, target xy, rms misalignment.

            # Exit loop if converged:
            if rms_misalignment <= MAX_MISALIGNMENT_FOR_CONVERGENCE:
                break

            # Get best similarity transform for each subarray (scikit-image.transform.SimilarityTransform):
            transforms = []
            for i_sa, sa in enumerate(self.subarrays):
                transform = skt.SimilarityTransform()  # populate this just below.
                current = np.array(current_locations[i_sa])
                transforms.append(transform.estimate(src=current, dst=target_locations))

            # Apply best similarity transform for each subarray (via scikit-image.transform.warp()):
            new_arrays = []
            for i_sa, sa in enumerate(self.subarrays):
                new_arrays.append(skt.warp(current_arrays[i_sa], transforms[i_sa], order=2, mode='edge'))
            current_arrays = new_arrays






















PLOTTING_FUNCTIONS____________________________________ = 0


def plot_subimages(figtitle, imagelist_object, plot_titles=None):
    arrays = [si.data for si in imagelist_object.subimages]
    if plot_titles is None:
        plot_titles = imagelist_object.filenames
    plot_arrays(figtitle, arrays, plot_titles)


def plot_subarrays(figtitle, subarraylist_object, plot_titles=None):
    arrays = [sa.array for sa in subarraylist_object.subarrays]
    if plot_titles is None:
        plot_titles = [sa.filename for sa in subarraylist_object.subarrays]
    plot_arrays(figtitle, arrays, plot_titles)


def plot_arrays(figtitle, arrays, plot_titles):
    """ Plot 4 arrays (each representing an image) per figure (page).
        Adapted from mpc.mp_planning.make_coverage_plots().
    :param figtitle: title for each figure. [string]
    :param arrays: list of arrays to plot. [list of numpy.ndarrays]
    :param plot_titles: list of titles for individual array plots. [list of strings]
    :return:
    """
    n_plots = len(arrays)
    n_cols, n_rows = 2, 2
    n_plots_per_figure = n_cols * n_rows
    n_figures = ceil(n_plots / n_plots_per_figure)  # count of pages of plots.
    plot_minimum = np.median([np.min(im) for im in arrays])
    plot_maximum = max([np.max(im) for im in arrays])
    norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(200.0))
    print('Plot (' + figtitle + '):', '{:.3f}'.format(plot_minimum), '{:.3f}'.format(plot_maximum))

    for i_figure in range(n_figures):
        n_plots_remaining = n_plots - (i_figure * n_plots_per_figure)
        n_plots_this_figure = min(n_plots_remaining, n_plots_per_figure)
        if n_plots_this_figure >= 1:
            fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(7, 4 * n_rows))
            # fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
            # fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
            fig.suptitle(figtitle + ' :: page ' + str(i_figure + 1))
            fig.canvas.set_window_title(figtitle + ' :: page ' + str(i_figure + 1))
            i_first = i_figure * n_plots_per_figure
            for i_plot in range(0, n_plots_this_figure):
                this_array = arrays[i_first + i_plot]
                this_title = plot_titles[i_first + i_plot]
                i_row, i_col = divmod(i_plot, n_cols)
                ax = axes[i_row, i_col]
                ax.set_title(this_title)
                ax.imshow(this_array, origin='upper', cmap='Greys', norm=norm)
            for i_plot_to_remove in range(n_plots_this_figure, n_plots_per_figure):
                i_col = i_plot_to_remove % n_cols
                i_row = int(floor(i_plot_to_remove / n_cols))
                ax = axes[i_row, i_col]
                ax.remove()
            plt.draw()
    plt.show()


# def plot_arrays(figtitle, arrays, plot_titles):
#     """ Plot 4 arrays (each representing an image) per figure (page).
#         Very closely adapted from mpc.mp_bulldozer.plot_images().
#     :param figtitle: title for each figure. [string]
#     :param arrays: list of arrays to plot. [list of numpy.ndarrays]
#     :param plot_titles: list of titles for individual array plots. [list of strings]
#     :return:
#     """
#     plots_per_figure = 4
#     plots_per_row = 2
#     rows_per_figure = plots_per_figure // plots_per_row
#     axes = None  # keep IDE happy.
#     plot_minimum = np.median([np.min(im) for im in arrays])
#     plot_maximum = max([np.max(im) for im in arrays])
#     # norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(1000.0))
#     norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(200.0))
#     print('Plot (' + figtitle + '):', '{:.3f}'.format(plot_minimum), '{:.3f}'.format(plot_maximum))
#     for i, im in enumerate(arrays):
#         _, i_plot_this_figure = divmod(i, plots_per_figure)
#         i_row, i_col = divmod(i_plot_this_figure, plots_per_row)
#         if i_plot_this_figure == 0:
#             fig, axes = plt.subplots(ncols=plots_per_row, nrows=rows_per_figure,
#                                      figsize=(7, 4 * rows_per_figure))
#             fig.suptitle(figtitle)
#         ax = axes[i_row, i_col]
#         ax.set_title(plot_titles[i])
#         im_plot = im.copy()
#         ax.imshow(im_plot, origin='upper', cmap='Greys', norm=norm)
#         if (i_plot_this_figure == plots_per_figure - 1) or (i == len(arrays) - 1):
#             plt.show()


SUPPORT_FUNCTIONS______________________________________ = 0


def apply_pixel_scale_factor(h, factor):
    """ Apply pixel scale correction factor to FITS header.
        Typically needed to adapt PinPoint's plate solution to image-averaged WCS without corrections.
    :param h: header without scale factor yet applied. [astropy.fits.io FITS header object]
    :param factor: the factor to apply, typically close to but not quite equal to 1. [float]
    :return: header with scale factor applied. [astropy.fits.io FITS header object]
    """
    cdelt1 = np.sign(h['CDELT1']) * ((abs(h['CDELT1']) + abs(h['CDELT2'])) / 2.0) * factor
    cdelt2 = np.sign(h['CDELT2']) * ((abs(h['CDELT1']) + abs(h['CDELT2'])) / 2.0) * factor
    cd11 = + cdelt1 * cos(h['CROTA2'] * RADIANS_PER_DEGREE)
    cd12 = - cdelt2 * sin(h['CROTA2'] * RADIANS_PER_DEGREE)
    cd21 = + cdelt1 * sin(h['CROTA2'] * RADIANS_PER_DEGREE)
    cd22 = + cdelt2 * cos(h['CROTA2'] * RADIANS_PER_DEGREE)
    h['CDELT1'] = cdelt1
    h['CDELT2'] = cdelt2
    h['CD1_1'] = cd11
    h['CD1_2'] = cd12
    h['CD2_1'] = cd21
    h['CD2_2'] = cd22
    return h


# class MP_Array:
#     """ Minimal masked array contstruct. Maybe not needed if MP_Image does the job. """
#     def __init__(self):
#         pass
#
#
# class MP_ArrayList:
#     """ List of minimal masked array contstructs. Maybe not needed if MP_ImageList does the job. """
#     def __init__(self):
#         pass

