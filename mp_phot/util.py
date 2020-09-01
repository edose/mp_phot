__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from math import floor, sqrt

# External packages:
import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from ccdproc import wcs_project, trim_image, Combiner
from photutils import make_source_mask, centroid_com

VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']


def get_mp_filenames(directory):
    """ Get only filenames in directory like MP_xxxxx.[ext], where [ext] is a legal FITS extension.
        The order of the return list is alphabetical (which may or may not be in time order). """
    all_filenames = pd.Series([e.name for e in os.scandir(directory) if e.is_file()])
    extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in all_filenames])
    is_fits = [ext.lower() in VALID_FITS_FILE_EXTENSIONS for ext in extensions]
    fits_filenames = all_filenames[is_fits]
    mp_filenames = [fn for fn in fits_filenames if fn.startswith('MP_')]
    mp_filenames.sort()
    return mp_filenames


def dict_from_directives_file(fullpath):
    """ For each directive line in file, return a value [string] or list of values [list of strings].
        Returned dict includes ONLY strings exactly as given without context, e.g., '16' for 16 and 'Yes'.
        Simple engine: no interpretation or checking: it is up to the calling routine to cast into needed
        types, to interpret these string values, and do any checking on their presence and/or values.
    :param fullpath: [string]
    :return: dict of key=directive [string]; value=value [string] or list of values [list of strings].
    """
    with open(fullpath) as f:
        lines = f.readlines()
    lines = [line for line in lines if line is not None]  # remove empty list elements
    lines = [line.split(";")[0] for line in lines]  # remove all comments
    lines = [line.strip() for line in lines]  # remove lead/trail blanks
    lines = [line for line in lines if line != '']  # remove empty lines
    lines = [line for line in lines if line.startswith('#')]  # keep only directive lines
    data_dict = dict()
    for line in lines:
        splitline = line.split(maxsplit=1)
        if len(splitline) != 2:
            print(' >>>>> ERROR: File', fullpath, 'cannot parse line', line)
        else:
            directive = (splitline[0])[1:].strip().upper()
            value = splitline[1].strip()
            previous_value = data_dict.get(directive, None)
            if previous_value is None:
                data_dict[directive] = value  #
            else:
                if not isinstance(previous_value, list):
                    data_dict[directive] = [data_dict[directive]]  # ensure previous value is a list.
                data_dict[directive].append(value)  # append list (not its indiv elements) into value list.
    return data_dict


IMAGE_AND_SUBIMAGE_UTILS_____________________________________ = 0


class Square:
    def __init__(self, image, x_center, y_center, square_radius, mask_radius=None):
        """ Copy of image slice, array if not cropped at edge. Centered on an integer pixel position.
            Added circular mask if requested (combined with original mask if present).
        :param image: source image. [masked image-like object, e.g., CCDData, but *not* MP_Image which is a
                   wrapper around CCDData.]
        :param x_center: in pixels. Rounded to nearest integer. [float]
        :param y_center: in pixels. Rounded to nearest integer. [float]
        :param square_radius: half-length of array's edge length less one, in pixels;
                   e.g., 10 to render a 21 x 21 array. Rounded down if not integral. [float]
        :param mask_radius: radius of center-aperture mask, or None/<=zero for no mask. [float or None]
        """
        # Make slice copy, with center and edges on integer pixel positions:
        self.x_center = int(round(x_center))  # x center position in parent array.
        self.y_center = int(round(y_center))  # y center position in parent array.
        self.radius = int(floor(square_radius))      # half of one less than expected Square edge length.
        self.mask_radius = mask_radius
        self.is_valid = not (self.x_center + self.radius < 0 or
                             self.y_center + self.radius < 0 or
                             self.x_center - self.radius > image.shape[1] or
                             self.y_center - self.radius > image.shape[0])
        self.x_low = max(0, self.x_center - self.radius)
        self.x_high = min(image.shape[1], self.x_center + self.radius)
        self.y_low = max(0, self.y_center - self.radius)
        self.y_high = min(image.shape[0], self.y_center + self.radius)

        cropped_image = image[self.y_low:self.y_high + 1,
                              self.x_low:self.x_high + 1].copy()  # ndarray & CCDData are [y,x]
        self.shape = cropped_image.shape
        expected_edge_size = 2 * square_radius + 1
        self.is_cropped = (self.shape != (expected_edge_size, expected_edge_size))

        # Ensure self.data is a simple numpy ndarray, even if a masked array or CCDData obj was passed in.
        if isinstance(cropped_image, np.ndarray):
            self.data = cropped_image  # numpy ndarray (nb: self.image.data is only a "memoryview" -- ugh).
        else:
            self.data = cropped_image.data  # numpy ndarray.
        self.parent = image  # access to parent image (rarely needed)

        # Ensure self.mask is present, usable, and a simple numpy ndarray of booleans:
        if isinstance(cropped_image, np.ndarray):
            self.mask = np.full_like(cropped_image.data, False, dtype=np.bool)  # image is ndarray.
        elif cropped_image.mask is None:
            self.mask = np.full_like(cropped_image.data, False, dtype=np.bool)  # image is CCDData, ma, etc.
        else:
            self.mask = cropped_image.mask.copy()  # if image is CCDData, masked array, etc.

        # Make radius mask if requested, then merge it with any pre-existing mask:
        if mask_radius is not None:
            if mask_radius > 0:
                circular_mask = np.fromfunction(lambda i, j: ((j - square_radius) ** 2 +
                                                              (i - square_radius) ** 2) > mask_radius ** 2,
                                                shape=self.shape)  # nb: True masks out that pixel.
                self.mask = self.mask | circular_mask

    def __str__(self):
        return 'Square object ' + str(self.shape[0]) + 'x' + str(self.shape[1]) +\
               ' at x,y = ' + '{:.3f}'.format(self.x_center) + ', ' + '{:.3f}'.format(self.y_center) + '.'

    def centroid(self, background_region='inverse'):
        """ Return position (parent image) of local flux centroid, background-subtracted if requested.
            No iteration: yields naive result.
        :param background_region: one of these [string or None]:
               'inverse' --> use the mask's inverse to calculate the background (normal case for stars etc);
               'all' --> use the whole Square to calculate the background (might have its uses);
               'mask' --> use the mask to calculate the background (probably not useful); or
               None or 'none' --> to not subtract background at all (by setting background to zero).
        :return: position (parent image) of local flux centroid. [2-tuple of floats]
        """
        if background_region is None:
            bkgd_adus = 0
        elif background_region.lower() == 'inverse':
            bkgd_adus, _ = calc_background_adus(self.data, ~self.mask)
        elif background_region.lower() == 'all':
            bkgd_adus, _ = calc_background_adus(self.data, None)
        elif background_region.lower() == 'mask':
            bkgd_adus, _ = calc_background_adus(self.data, self.mask)
        elif background_region.lower() == 'none':
            bkgd_adus = 0
        else:
            print(' >>>>> WARNING: centroid() has background_region of',
                  str(background_region), 'which is not a legal value. Using value of \'all\'.')
            bkgd_adus, _ = calc_background_adus(self.data, None)
        data = self.data - bkgd_adus
        x_square, y_square = centroid_com(data=data, mask=self.mask)
        x_centroid = self.x_low + x_square
        y_centroid = self.y_low + y_square
        return x_centroid, y_centroid  # relative to parent image (0,0).

    def recentroid(self, background_region='inverse', max_iterations=3):
        """ Iterative search of centroid, valid even if square has to be resliced because centroid
            is farther than 1/2 pixel from expected position (common).
        :param background_region: same as for centroid(), i.e., one of these [string or None]:
            'inverse' --> use the mask's inverse to calculate the background (normal case for stars etc);
            'all' --> use the whole Square to calculate the background (might have its uses);
            'mask' --> use the mask to calculate the background (probably not useful); or
            None or 'none' --> to not subtract background at all (by setting background to zero).
        :param max_iterations: number of iterations allowed. [int]
        :return: x_centroid, y_centroid. [2-tuple of floats]
        """
        square = self
        x, y = None, None  # keep IDE happy
        for i in range(max_iterations):
            x, y = square.centroid(background_region=background_region)
            need_new_square = (abs(x - square.x_center) > 0.5 or abs(x - square.x_center) > 0.5)
            if need_new_square and i < max_iterations - 1:
                square = Square(square.parent, x, y, self.radius, self.mask_radius)
            else:
                break
        return x, y  # relative to parent image (0,0).


def calc_background_adus(data, mask=None, invert_mask=False):
    """ Calculate the sigma-clipped median background of a (masked)array (or array slice).
    :param data: array of pixels. [2-D ndarray of floats]
    :param mask: mask array, True=mask. [None, or 2-D nadarray of bool]
    :param invert_mask: True to invert the image's mask before usage
    :return: background adu level (flux per pixel), or None if no available pixels. [float]
    """
    if mask is not None:
        this_mask = mask.copy()
        if invert_mask:
            this_mask = ~this_mask
        try:
            bkgd_mask = make_source_mask(data, mask=this_mask, nsigma=2, npixels=5,
                                         filter_fwhm=2, dilate_size=11)
        except ValueError:
            bkgd_mask = np.full_like(this_mask, False, np.bool)
            print(' >>>>> WARNING: background mask could not be made; blank mask used.')
        bkgd_mask = bkgd_mask | this_mask
    else:
        bkgd_mask = make_source_mask(data, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
    _, median, std = sigma_clipped_stats(data, sigma=3.0, mask=bkgd_mask)
    return median, std


def make_pill_mask(mask_shape, xa, ya, xb, yb, radius):
    """ Construct a mask array for MP in motion: unmask only those pixels within radius pixels of
        any point in line segment from (xa,ya) to (xb,yb).
    :param mask_shape: [y,x] shape of new mask. [2-tuple of floats]
    :param xa: [float]
    :param ya: "
    :param xb: "
    :param yb: "
    :param radius: width of unmasked region. [float]
    :return: np.ndarray with False
    """
    dx = xb - xa
    dy = yb - ya
    distance_motion = sqrt(dx**2 + dy**2)

    # Unmask up to radius distance from each endpoint:
    circle_a_mask = np.fromfunction(lambda i, j: ((i - ya) ** 2 + (j - xa) ** 2) > (radius ** 2),
                                    shape=mask_shape)
    circle_b_mask = np.fromfunction(lambda i, j: ((i - yb) ** 2 + (j - xb) ** 2) > (radius ** 2),
                                    shape=mask_shape)

    # Nested function:
    def distance_to_line(x0, y0, x1, y1, x2, y2, dist12=None):
        if dist12 is None:
            dist12 = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / dist12
        return distance

    # Mask outside max distance from (xa,ya)-(xb,yb) line segment:
    rectangle_submask_1 = np.fromfunction(lambda i, j:
                                          distance_to_line(j, i, xa, ya, xb, yb, distance_motion) > radius,
                                          shape=mask_shape)

    # Mask ahead of or behind MP motion line segment:
    dx_left = dy
    dy_left = -dx
    dx_right = -dy
    dy_right = dx
    x_center = (xa + xb) / 2.0
    y_center = (ya + yb) / 2.0
    x_left = x_center + dx_left
    y_left = y_center + dy_left
    x_right = x_center + dx_right
    y_right = y_center + dy_right
    distance_perpendicular = sqrt((x_right - x_left)**2 + (y_right - y_left)**2)  # prob = distance_motion
    rectangle_submask_2 = np.fromfunction(lambda i, j:
                                          distance_to_line(j, i, x_left, y_left, x_right, y_right,
                                                           distance_perpendicular) > distance_motion / 2.0,
                                          shape=mask_shape)
    # Combine masks and return:
    rectangle_mask = np.logical_or(rectangle_submask_1, rectangle_submask_2)  # intersection of False.
    circles_mask = np.logical_and(circle_a_mask, circle_b_mask)  # union of False.
    mask = np.logical_and(circles_mask, rectangle_mask)          # "
    return mask


PROBABLY_NOT_USED_____________________________________ = 0


def shift_2d_array(arr, dx, dy, fill_value=np.nan):
    """ Integer-pixel internal shift of data within a numpy array.
        Clever, and tests well, but I don't think we can use it in mp_phot."""
    new_arr = np.full_like(arr, fill_value)
    if dx == 0 and dy == 0:
        return arr.copy()
    if dx == 0 and dy > 0:
        new_arr[dy:, :] = arr[:-dy, :]
    elif dx == 0 and dy < 0:
        new_arr[:dy, :] = arr[-dy:, :]
    elif dx > 0 and dy == 0:
        new_arr[:, dx:] = arr[:, :-dx]
    elif dx < 0 and dy == 0:
        new_arr[:, :dx] = arr[:, -dx:]
    elif dx > 0 and dy > 0:
        new_arr[dy:, dx:] = arr[:-dy, :-dx]
    elif dx > 0 and dy < 0:
        new_arr[:dy, dx:] = arr[-dy:, :-dx]
    elif dx < 0 and dy > 0:
        new_arr[dy:, :dx] = arr[:-dy, -dx:]
    else:
        new_arr[:dy, :dx] = arr[-dy:, -dx:]
    return new_arr

