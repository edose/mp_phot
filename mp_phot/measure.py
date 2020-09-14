__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from math import pi, cos, sin, floor, ceil, sqrt, log10, log

# External packages:
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as apyfits
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
# from astropy.modeling import models, fitting
from astropy.modeling.models import Gaussian2D
from astropy.convolution import convolve
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from ccdproc import trim_image, wcs_project, Combiner
from photutils import SplitCosineBellWindow, CosineBellWindow, \
    data_properties, centroid_com, create_matching_kernel, make_source_mask
import skimage.transform as skt
import statsmodels.api as sma

# From this package:
from mp_phot import util

PINPOINT_PLTSOLVD_TEXT = 'Plate has been solved by PinPoint'
PIXEL_FACTOR_HISTORY_TEXT = 'Pixel scale factor applied by apply_pixel_scale_factor().'
RADIANS_PER_DEGREE = pi / 180.0
MAX_MISALIGNMENT_FOR_CONVERGENCE = 5.0  # in millipixels
FWHM_PER_SIGMA = 2.0 * sqrt(2.0 * log(2))  # ca. 2.35482


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
        self.mp_locations_2refs = mp_locations  # originally read locations from 2 reference FITS.
        self.mp_radecs = []              # placeholder
        self.mp_start_radecs = []
        self.mp_end_radecs = []
        self.mp_locations_all = []       # "
        self.mp_start_locations = []
        self.mp_end_locations = []
        self.settings = settings

        self.subimages = []  # placeholder
        self.subimage_ref_star_locations = []  # will be a list of lists of [x,y] lists.
        self.subimage_mp_locations = []
        self.subimage_mp_start_locations = []
        self.subimage_mp_end_locations = []
        self.subimage_sky_adus = []

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
            self.ref_star_radecs.append([ra, dec])
            # self.ref_star_radecs.append((ra, dec))  # x,y & ra,dec tuples -> lists 2020-08-27.

    def calc_mp_radecs(self):
        """ Use WCS (plate solution) to find MP's RA,Dec in *all* images, given MP *center* (x,y).
            Then calculate exposure-start and -stop RA,Dec for each image.
        """
        radius = 8 * self.settings['FWHM_NOMINAL']  # 1/2 size of array including background estimation.
        mask_radius = radius / 2                    # radius of area for flux centroid estimation.
        # Get MP center RA,Dec for 2 user-specified images:
        jd_mids, ra_values, dec_values = [], [], []
        for (filename, x0, y0) in self.mp_locations_2refs[:2]:
            this_image = self[filename].image
            ra, dec = tuple(this_image.wcs.all_pix2world([list((x0, y0))], 0)[0])
            jd_mids.append(self[filename].jd_mid)
            ra_values.append(ra)
            dec_values.append(dec)

        # Compute mid-exposure RA,Dec and x,y location of MP for each image:
        span_seconds = (jd_mids[1] - jd_mids[0]) * 24 * 3600  # timespan between mid JD of 2 MP images.
        ra_per_second = (ra_values[1] - ra_values[0]) / span_seconds
        dec_per_second = (dec_values[1] - dec_values[0]) / span_seconds
        for mp_image in self.mp_images:
            dt = (mp_image.jd_mid - jd_mids[0]) * 24 * 3600
            ra = ra_values[0] + dt * ra_per_second
            dec = dec_values[0] + dt * dec_per_second
            x, y = tuple(mp_image.image.wcs.all_world2pix([[ra, dec]], 0, ra_dec_order=True)[0])
            self.mp_radecs.append([ra, dec])
            self.mp_locations_all.append([x, y])
        # Compute exposure-start and -end RA,Dec and x,y location of MP for each image:
        for mp_image in self.mp_images:
            dt_start = (mp_image.jd_mid - jd_mids[0]) * 24 * 3600 - mp_image.exposure / 2.0
            ra_start = ra_values[0] + dt_start * ra_per_second
            dec_start = dec_values[0] + dt_start * dec_per_second
            x_start, y_start = tuple(mp_image.image.wcs.all_world2pix([[ra_start, dec_start]],
                                                                      0, ra_dec_order=True)[0])
            self.mp_start_radecs.append([ra_start, dec_start])
            self.mp_start_locations.append([x_start, y_start])

            dt_end = dt_start + mp_image.exposure
            ra_end = ra_values[0] + dt_end * ra_per_second
            dec_end = dec_values[0] + dt_end * dec_per_second
            x_end, y_end = tuple(mp_image.image.wcs.all_world2pix([[ra_end, dec_end]],
                                                                  0, ra_dec_order=True)[0])
            self.mp_end_radecs.append([ra_end, dec_end])
            self.mp_end_locations.append([x_end, y_end])

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
            y_values = y_ref_star_raw[i].copy()  # "
            x_values.append(self.mp_locations_all[i][0])
            y_values.append(self.mp_locations_all[i][1])
            x_min.append(min(x_values) - x_offsets[i])
            x_max.append(max(x_values) - x_offsets[i])
            y_min.append(min(y_values) - y_offsets[i])
            y_max.append(max(y_values) - y_offsets[i])

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

        # Get background ADUs for each subimage (before WCS alignment):
        for i, si in enumerate(self.subimages):
            median_adu, _ = util.calc_background_adus(si.data)
            self.subimage_sky_adus.append(median_adu)

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
                this_subimage_ref_star_locations.append([xc, yc])
                # x,y & ra,dec tuples -> lists 2020-08-27.
                # this_subimage_ref_star_locations.append((xc, yc))
            self.subimage_ref_star_locations.append(this_subimage_ref_star_locations)

        # Get all subimage MP locations:
        for i, si in enumerate(self.subimages):
            radec = self.mp_radecs[i]
            x, y = tuple(si.wcs.all_world2pix([radec], 0, ra_dec_order=True)[0])
            self.subimage_mp_locations.append([x, y])
            # self.subimage_mp_locations.append((x, y))  # x,y & ra,dec tuples -> lists 2020-08-27.
            radec_start = self.mp_start_radecs[i]
            x_start, y_start = tuple(si.wcs.all_world2pix([radec_start], 0, ra_dec_order=True)[0])
            self.subimage_mp_start_locations.append([x_start, y_start])
            # x,y & ra,dec tuples -> lists 2020-08-27.
            # self.subimage_mp_start_locations.append((x_start, y_start))
            radec_end = self.mp_end_radecs[i]
            x_end, y_end = tuple(si.wcs.all_world2pix([radec_end], 0, ra_dec_order=True)[0])
            self.subimage_mp_end_locations.append([x_end, y_end])
            # x,y & ra,dec tuples -> lists 2020-08-27.
            # self.subimage_mp_end_locations.append((x_end, y_end))

        # # TEST ONLY: test subimage MP locations (ex earliest and latest) for time-interpolation accuracy:
        # xy_earliest = self.subimage_mp_locations[0]
        # xy_latest = self.subimage_mp_locations[-1]
        # for i, mploc in enumerate(self.subimage_mp_locations[1:-1]):
        #     jd_fraction = (self.jd_mids[i] - self.jd_mids[0]) / (self.jd_mids[-1] - self.jd_mids[0])
        #     x_expected = xy_earliest[0] + jd_fraction * (xy_latest[0] - xy_earliest[0])
        #     y_expected = xy_earliest[1] + jd_fraction * (xy_latest[1] - xy_earliest[1])
        #     x_actual = self.subimage_mp_locations[i][0]
        #     y_actual = self.subimage_mp_locations[i][1]
        #     distance = sqrt((x_actual - x_expected)**2 + (y_actual - y_expected)**2)
        #     print('{:3d}'.format(i) + ':',
        #           '  expected=', '{:8.3f}'.format(x_expected), '{:8.3f}'.format(x_expected),
        #           '  actual=', '{:8.3f}'.format(x_actual), '{:8.3f}'.format(y_actual),
        #           '  dist=', '{:8.3f}'.format(distance), 'pixels.')

    def make_subarrays(self):
        subarrays = []
        for i, si in enumerate(self.subimages):
            subarray = Subarray(filename=self.filenames[i],
                                array=si.data,
                                mask=si.mask,
                                jd_mid=self.jd_mids[i],
                                exposure=self.jd_mids[i],
                                original_image_shape=self.mp_images[i].image.shape,
                                original_mp_location=self.mp_locations_all[i],
                                original_sky_adu=self.subimage_sky_adus[i],
                                ref_star_locations=self.subimage_ref_star_locations[i],
                                mp_location=self.subimage_mp_locations[i],
                                mp_start_location=self.subimage_mp_start_locations[i],
                                mp_end_location=self.subimage_mp_end_locations[i])
            subarrays.append(subarray)
        subarray_list = SubarrayList(subarrays=subarrays,
                                     mp_id=self.mp_id,
                                     an=self.an,
                                     filter=self.filter,
                                     settings=self.settings)
        return subarray_list


class Subarray:
    def __init__(self, filename, array, mask, jd_mid, exposure, original_image_shape, original_mp_location,
                 original_sky_adu, ref_star_locations, mp_location, mp_start_location, mp_end_location):
        """ Hold one subarray and matching mask, for an image segment.  For late-stage processing.
            NO WCS, no RA,Dec data for MP or ref stars.
        :param filename: [string]
        :param array: [ndarray of floats]
        :param mask: [ndarray of bools]
        :param jd_mid: [float]
        :param exposure: [float]
        :param original_image_shape: [2-tuple of floats]
        :param original_mp_location: [2-tuple of floats]
        :param original_sky_adu: [float]
        :param ref_star_locations: [list of floats]
        :param mp_location: [float]
        :param mp_start_location: [float]
        :param mp_end_location: [float]
        """
        self.filename = filename
        self.array = array.copy()
        if mask is None:
            self.mask = np.full_like(self.array, False, np.bool)
        else:
            self.mask = mask.copy()
        self.jd_mid = jd_mid
        self.exposure = exposure
        self.original_image_shape = original_image_shape
        self.original_mp_location = original_mp_location
        self.original_sky_adu = original_sky_adu
        self.ref_star_locations = ref_star_locations.copy()
        self.mp_location = mp_location
        self.mp_start_location = mp_start_location
        self.mp_end_location = mp_end_location
        self.ref_star_squares = []   # placeholder, will be a list of util.Square objects.
        self.fwhm = None
        self.matching_kernel = None             # placeholder, will be a 2-d np.ndarray
        self.convolved_array = None             # placeholder, will be a 2-d np.ndarray
        self.realigned_array = None             # placeholder, will be a 2-d np.ndarray
        self.realigned_ref_star_locations = []  # placeholder, will be a 2-d np.ndarray
        self.realigned_mp_location = None
        self.realigned_mp_start_location = None
        self.realigned_mp_end_location = None
        self.realigned_mp_mask = None            # placeholder, float. Mask polarity: to reveal MP
        self.realigned_mp_only_array = None      # placeholder, will be a 2-d np.ndarray
        self.mp_flux = None                      # placeholder, float, best aperture-phot net flux (ADUs)
        self.mp_sigma = None                     # placeholder, float, best aperture-phot flux sigma (ADUs).

    def __str__(self):
        return 'Subarray object from ' + self.filename + ' of shape ' + str(self.array.shape)


class SubarrayList:
    def __init__(self, subarrays, mp_id, an, filter, settings):
        self.subarrays = subarrays
        self.mp_id = mp_id
        self.an = an
        self.filter = filter
        self.settings = settings
        self.nominal_sigma = None
        self.best_bkgd_array = None

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
            Little or no shift in light source centroids (kernels centered).
        :return: None. Subarraylist.subarray objects are populated with matching kernels.
                 [list of lists of numpy.ndarrays]
        """
        # Make ref star PSF arrays (very small kernel-sized images) [save as util.Square objects]:
        radius = 5 * self.settings['FWHM_NOMINAL']
        mask_radius = 2.5 * self.settings['FWHM_NOMINAL']
        for sa in self.subarrays:
            subarray_psfs = []  # for this subarray (all ref stars).
            for x_center, y_center in sa.ref_star_locations:
                # Adding a mask_radius makes this work more cleanly:
                square = util.Square(sa.array.astype(np.float), x_center, y_center, radius, mask_radius)
                bkgd_adus, _ = util.calc_background_adus(square.data)
                bkdg_subtracted_array = square.data - bkgd_adus
                # Vignette bkgd-subtracted array (reduce noise far from center, in lieu of sharp mask):
                # Nullify mask after getting bkgd; window array will work better:
                square.mask = np.full_like(square.mask, False, np.bool)
                window_array = (SplitCosineBellWindow(alpha=0.4, beta=0.6))(square.shape)
                raw_kernel = window_array * bkdg_subtracted_array
                square.data = raw_kernel / np.sum(raw_kernel)  # normalized kernel.
                sa.ref_star_squares.append(square)
                subarray_psfs.append(square)
            sa.ref_star_squares = subarray_psfs

        # Calculate smallest appropriate sigma for target 2_D Gaussian PSF (everything scales from this):
        sigma_list = []  # list of median sigma (one element per subarray).
        for sa in self.subarrays:
            sa_sigmas = []
            fwhm_values = []
            for rss in sa.ref_star_squares:
                dps = data_properties(rss.data)
                sa_sigmas.append(dps.semimajor_axis_sigma.value)
                # print('   ',  '{:.2f}'.format(dps.xcentroid.value),
                #               '{:.2f}'.format(dps.ycentroid.value),
                #               '{:.3f}'.format(dps.semimajor_axis_sigma.value))
            sa_sigma = np.median(np.array(sa_sigmas))
            sigma_list.append(sa_sigma)
            sa.fwhm = sa_sigma * FWHM_PER_SIGMA
        max_sigma = max(sigma_list)
        target_sigma = 1.0 * max_sigma  # use max sigma for little/no sharpening when convoluting.

        # Make matching kernels:
        for i_sa, sa in enumerate(self.subarrays):
            sa_matching_kernels = []
            for i_rs, rs in enumerate(sa.ref_star_locations):
                this_psf = sa.ref_star_squares[i_rs]
                edge_length = this_psf.shape[0]
                y, x = np.mgrid[0:edge_length, 0:edge_length]
                # Relative to center; bkgd_region None to avoid masking away all pixels.
                x_center_parent, y_center_parent = this_psf.recentroid(background_region=None)
                x_center0 = x_center_parent - this_psf.x_low
                y_center0 = y_center_parent - this_psf.y_low
                # Make target PSFs (2-D Gaussian):
                gaussian = Gaussian2D(1, x_center0, y_center0, target_sigma, target_sigma)  # function
                target_psf = gaussian(x, y)  # ndarray
                target_psf /= np.sum(target_psf)  # ensure normalized to sum 1.
                # Make matching_kernel (to make this_psf match target_psf):
                matching_kernel = create_matching_kernel(this_psf.data, target_psf,
                                                         CosineBellWindow(alpha=0.35))
                sa_matching_kernels.append(matching_kernel)
            sa.matching_kernel = np.mean(np.stack(sa_matching_kernels), axis=0)

    def convolve_subarrays(self):
        """ Convolve subarrays with matching kernels to render new subarrays with very nearly Gaussian
            ref star PSFs with very nearly uniform sigma.
        :return: None. Subarraylist.subarray objects are populated with convolved arrays.
        """
        new_subarrays = []
        for i_sa, sa in enumerate(self.subarrays):
            sa.convolved_array = convolve(sa.array.copy(), sa.matching_kernel, boundary='extend')

    # def convolve_full_arrays(self, array_list):
    #     """ This is only for feasibility timing. """
    #     from datetime import datetime
    #     print('before full convolution:', datetime.now())
    #     for i, arr in enumerate(array_list):
    #         print('\nstart convolving full array', str(i))
    #         conv_arr = convolve(arr.copy(), self.subarrays[i].matching_kernel, boundary='extend')
    #         print('finish convolving full array', str(i))
    #     print('after full convolution: ', datetime.now())

    def realign(self, max_iterations=4):
        """ Iteratively improve array alignments to millipixel precision.
            Each loop: (1) get best similarity transform (rotation+translation+scale)
            from ref star positions vs desired positions, (2) perform transform.
        :param max_iterations: [int]
        :return: None.
        """
        # Set target ref star pixel x,y locations as the mean recentroided location, across subarrays:
        radius = 5 * self.settings['FWHM_NOMINAL']
        target_ref_star_locations = []
        ref_star_sigmas = []
        for i_rs in range(len(self.subarrays[0].ref_star_locations)):
            x_list, y_list = [], []
            for sa in self.subarrays:
                square = util.Square(sa.convolved_array,
                                     sa.ref_star_locations[i_rs][0],
                                     sa.ref_star_locations[i_rs][1], radius, mask_radius=radius / 2)
                x, y = square.recentroid()
                x_list.append(x)
                y_list.append(y)
                bkgd_adus, _ = util.calc_background_adus(square.data, square.mask, invert_mask=True)
                dps = data_properties(square.data - bkgd_adus, square.mask)
                sigma = dps.semimajor_axis_sigma.value  # in pixels.
                # print(str(i_rs), '   sigma: ', '{:3f}'.format(sigma))
                ref_star_sigmas.append(sigma)
            target_ref_star_locations.append([np.mean(x_list), np.mean(y_list)])
        target_ref_star_locations = np.array(target_ref_star_locations)
        nominal_sigma = max(ref_star_sigmas)  # for use in settings masks, below.

        # Initialize the best/current subarrays and mp_locations:
        current_arrays = [sa.convolved_array for sa in self.subarrays]
        current_ref_star_locations = [sa.ref_star_locations for sa in self.subarrays]
        current_mp_locations = [sa.mp_location for sa in self.subarrays]
        current_mp_start_locations = [sa.mp_start_location for sa in self.subarrays]
        current_mp_end_locations = [sa.mp_end_location for sa in self.subarrays]

        # Nested function (will be called below, at least twice):
        def estimate_rms_misalignment(current_arrays, current_ref_star_locations,
                                      target_ref_star_locations, mask_sigma=None, do_print=False):
            total_squared_misalignment = 0.0  # in millipixels.
            n_centers = 0
            recentroided_ref_star_locations = []
            for i_sa, sa in enumerate(current_arrays):
                sa_recentroided_ref_star_locations = []
                for i_rs in range(len(current_ref_star_locations[i_sa])):
                    square = util.Square(current_arrays[i_sa],
                                         current_ref_star_locations[i_sa][i_rs][0],
                                         current_ref_star_locations[i_sa][i_rs][1],
                                         radius, mask_radius=3*mask_sigma)
                    x, y = square.recentroid()
                    sa_recentroided_ref_star_locations.append([x, y])
                    # x,y & ra,dec tuples -> lists 2020-08-27.
                    # recentroided_ref_star_locations.append((x, y))
                    x_target, y_target = target_ref_star_locations[i_rs]
                    total_squared_misalignment += ((x - x_target)**2 + (y - y_target)**2)
                    n_centers += 1
                recentroided_ref_star_locations.append(sa_recentroided_ref_star_locations)
            rms_misalignment = 1000.0 * sqrt(total_squared_misalignment / n_centers)  # in millipixels.

            # Print  target xy, current xy, recentroided (new) xy.
            print('RMS Misalignment =', '{:.1f}'.format(rms_misalignment), 'millipixels.')
            for i_sa, sa in enumerate(self.subarrays):
                # print('    Subarray', i_sa)
                for i_rs in range(len(current_ref_star_locations[i_sa])):
                    target_locs = target_ref_star_locations[i_rs]
                    current_locs = current_ref_star_locations[i_sa][i_rs]
                    recentroided_locs = recentroided_ref_star_locations[i_sa][i_rs]
                    # print('        target:', '{:.3f}'.format(target_locs[0]),
                    #                          '{:.3f}'.format(target_locs[1]),
                    #                          '  from {:.3f}'.format(current_locs[0]),
                    #                          ' {:.3f}'.format(current_locs[1]),
                    #                          '  to {:.3f}'.format(recentroided_locs[0]),
                    #                          ' {:.3f}'.format(recentroided_locs[1]))
            return rms_misalignment, recentroided_ref_star_locations

        rms_misalignment, current_ref_star_locations = \
            estimate_rms_misalignment(current_arrays, current_ref_star_locations,
                                      target_ref_star_locations, mask_sigma=nominal_sigma,
                                      do_print=True)

        # ***** MAIN REALIGNMENT LOOP *******************************************:
        for i_loop in range(max_iterations):
            if rms_misalignment <= MAX_MISALIGNMENT_FOR_CONVERGENCE:
                print('Converged and exit, i_loop=', str(i_loop))
                break

            # TODO: List transforms for each subarray, then combine transforms and apply only after loop.
            # TODO: combined transforms should be applied to: all mp_locations.
            # Make best similarity transform for each subarray (scikit-image.transform.SimilarityTransform):
            transforms = []
            for i_sa, sa in enumerate(self.subarrays):
                sa_transform = skt.estimate_transform(ttype='similarity',
                                                      src=np.array(current_ref_star_locations[i_sa]),
                                                      dst=target_ref_star_locations)
                transforms.append(sa_transform)

            # Apply best similarity transform for each subarray (via scikit-image.transform.warp()):
            new_arrays = []
            new_mp_locations, new_mp_start_locations, new_mp_end_locations = [], [], []
            for i_sa, sa in enumerate(self.subarrays):
                this_transform = transforms[i_sa]
                # Realign all data arrays:
                new_arrays.append(skt.warp(current_arrays[i_sa], inverse_map=this_transform.inverse,
                                           order=1, mode='edge'))  # biquad has skimage bug (as of 2020-09).
                # Transform (realign) all MP locations:
                new_mp_locations.append(this_transform(current_mp_locations[i_sa])[0])
                new_mp_start_locations.append(this_transform(current_mp_start_locations[i_sa])[0])
                new_mp_end_locations.append(this_transform(current_mp_end_locations[i_sa])[0])

            # Save new arrays and MP locations as current (to use in next loop cycle, or as final values):
            # NB: new ref star locations come from estimate_rms_misalignment(), below.
            current_arrays = new_arrays
            current_mp_locations = new_mp_locations
            current_mp_start_locations = new_mp_start_locations
            current_mp_end_locations = new_mp_end_locations

            rms_misalignment, current_ref_star_locations = \
                estimate_rms_misalignment(current_arrays, current_ref_star_locations,
                                          target_ref_star_locations, mask_sigma=nominal_sigma,
                                          do_print=True)

        # ***** END of MAIN REALIGNMENT LOOP ************************************:

        # When loop finished, save best realigned data in objects:
        for i_sa, sa in enumerate(self.subarrays):
            sa.realigned_array = current_arrays[i_sa]
            sa.realigned_ref_star_locations = current_ref_star_locations[i_sa]
            sa.realigned_mp_location = current_mp_locations[i_sa]
            sa.realigned_mp_start_location = current_mp_start_locations[i_sa]
            sa.realigned_mp_end_location = current_mp_end_locations[i_sa]
        self.nominal_sigma = nominal_sigma

    def make_best_bkgd_array(self):
        """ Make reference (MP-masked, background-subtracted) averaged subarray.
            This is what this sliver of sky would look like, on average, if the MP were not there at all.
        :return: None
        """
        ccddata_objects = []
        for i_sa, sa in enumerate(self.subarrays):
            bkgd_adus, _ = util.calc_background_adus(sa.realigned_array)
            bkgd_subtr_array = sa.realigned_array - bkgd_adus
            sa.realigned_mp_mask = util.make_pill_mask(bkgd_subtr_array.shape,
                                                       sa.realigned_mp_start_location[0],
                                                       sa.realigned_mp_start_location[1],
                                                       sa.realigned_mp_end_location[0],
                                                       sa.realigned_mp_end_location[1],
                                                       5 * self.nominal_sigma)  # polarity: reveals MP.
            ccddata_objects.append(CCDData(data=bkgd_subtr_array,
                                           mask=~sa.realigned_mp_mask,   # mask polarity: hides MP.
                                           unit='adu'))
        averaged_ccddata = Combiner(ccddata_objects).average_combine()
        n_masked_out = np.sum(averaged_ccddata.mask)
        if n_masked_out > 0:
            print(' >>>>> WARNING: averaged_subarray has', str(n_masked_out), 'pixels masked out.')
        self.best_bkgd_array = averaged_ccddata.data

    def make_mp_only_subarrays(self):
        """ For each realigned, background-subtracted, MP-masked subarray, find (using ordinary least-sq):
            source_factor (amount of star flux relative to averaged subarray source fluxes), and
            background_offset (uniform background ADUs, relative to averaged subarray flat background).
        :return: [None]
        """
        # Decompose subarrays into source component and background component (extract coefficients):
        # fit = fitting.LinearLSQFitter()
        # simple_linear = models.Linear1D()
        indep_var_unmasked = np.ravel(self.best_bkgd_array)
        source_factors, background_offsets = [], []
        for i, sa in enumerate(self.subarrays):
            dep_var_unmasked = np.ravel(sa.realigned_array)
            # NB: Polarities in next line: to_keep=True reveals, numpy mask=True hides.
            to_keep = np.ravel(sa.realigned_mp_mask)
            indep_var = indep_var_unmasked[to_keep]
            dep_var = dep_var_unmasked[to_keep]
            # Adding backgrounds' x- and y-gradients (next lines) as fit parameters doesn't seem to help.
            # y_grid, x_grid = np.mgrid[0:self.best_bkgd_array.shape[0], 0:self.best_bkgd_array.shape[1]]
            # y_mean = (y_grid.shape[0] - 1) / 2.0
            # x_mean = (x_grid.shape[1] - 1) / 2.0
            # y_1024 = (np.ravel(y_grid)[to_keep] - y_mean) / 1024
            # x_1024 = (np.ravel(x_grid)[to_keep] - x_mean) / 1024
            # indep_var = np.column_stack((indep_var, x_1024, y_1024))
            indep_var = sma.add_constant(data=indep_var)  # statsmodels' weird way of "adding" an intercept.
            results = sma.OLS(dep_var, indep_var).fit()
            background_offsets.append(results.params[0])
            source_factors.append(results.params[1])
            print('\ndecomp:', str(i), sa.filename,
                  '  bkgd_offset=', '{:.3f}'.format(background_offsets[i]),
                  '({:.3f})'.format(results.bse[0]),
                  '  source_factor=', '{:.4f}'.format(source_factors[i]),
                  '({:.4f})'.format(results.bse[0]),
                  '  R2=', '{:.6f}'.format(results.rsquared))
            # print('     x_1024=', '({:.6f})'.format(results.params[2]), '({:.6f})'.format(results.bse[2]),
            #       '     y_1024=', '({:.6f})'.format(results.params[3]), '({:.6f})'.format(results.bse[3]))

        # Using coefficients, generate nominal mp_free_subarrays:
        for (sf, bo, sa) in zip(source_factors, background_offsets, self.subarrays):
            fitted_whole_background = bo + sf * self.best_bkgd_array
            sa.realigned_mp_only_array = sa.realigned_array - fitted_whole_background

    def do_mp_aperture_photometry(self):
        """  Do aperture photometry on best MP-only subarrays, using pill masks previously saved.
        :return: [None]
        """
        for sa in self.subarrays:
            this_masked_array = np.ma.array(data=sa.realigned_mp_only_array,
                                            mask=sa.realigned_mp_mask)  # whole subarray.
            mp_square = util.Square(this_masked_array,
                                    sa.realigned_mp_location[0],
                                    sa.realigned_mp_location[1],
                                    10 * self.nominal_sigma)
            masked_square = np.ma.array(data=mp_square.data,
                                        mask=mp_square.mask)  # small square slice of subarray.
            mp_aperture_flux = np.sum(masked_square)       # from masked sum.
            mp_aperture_area = np.sum(masked_square.mask)  # pixel count.
            bkgd_median, bkgd_std = util.calc_background_adus(data=masked_square.data,
                                                              mask=masked_square.mask,
                                                              invert_mask=True)  # True to hide MP.
            sa.mp_flux = mp_aperture_flux - mp_aperture_area * bkgd_median
            sigma2_ap = self.settings['CCD_GAIN'] * (mp_aperture_flux + mp_aperture_area * bkgd_median)
            sigma2_bkgd = mp_aperture_area * bkgd_std**2
            sa.mp_sigma = sqrt(sigma2_ap + sigma2_bkgd)

    def make_df_mp_only(self):
        """ Render df_mp_only.
        :return: [None]
        """
        # Make lists to hold data for dataframe columns:
        dict_list = []
        for sa in self.subarrays:
            x_original_image_center = sa.original_image_shape[1] / 2.0
            y_original_image_center = sa.original_image_shape[0] / 2.0
            x_1024 = (sa.original_mp_location[0] - x_original_image_center) / 1024.0
            y_1024 = (sa.original_mp_location[1] - y_original_image_center) / 1024.0
            sa_dict = {
                'FITSfile': sa.filename,
                'SourceID': self.mp_id,
                'Type': 'MP',
                'InstMag': -2.5 * log10(sa.mp_flux / sa.exposure),
                'InstMagSigma': -2.5 * log10(sa.mp_sigma / sa.mp_flux),
                'FWHM': sa.fwhm,
                'SkyADU': sa.original_sky_adu,
                'XCentroid': sa.original_mp_location[0],
                'YCentroid': sa.original_mp_location[1],
                'X1024': x_1024,
                'Y1024': y_1024,
                'Vignette': x_1024**2 + y_1024**2
                }
            dict_list.append(sa_dict)
        df_mp_only = pd.DataFrame(data=dict_list)
        return df_mp_only


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
    print('Plot (' + figtitle + '): adu range=', '{:.3f}'.format(plot_minimum), 'to {:.3f}'.format(plot_maximum))

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
    # plt.show()


def plot_one_array(figtitle, array):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    source_mask = make_source_mask(array, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
    _, median, _ = sigma_clipped_stats(array, sigma=3.0, mask=source_mask)
    plot_minimum = median
    plot_maximum = np.max(array)
    norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(250.0))
    ax.imshow(array, origin='upper', cmap='Greys', norm=norm)
    fig.suptitle(figtitle)
    # plt.show()


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

