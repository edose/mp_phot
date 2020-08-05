__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import numpy as np
import pandas as pd
from astropy.nddata import CCDData

# From this package:
from .util import get_mp_filenames


def measure_mp():
    pass


def measure_comps():
    pass


class WholeImageList:
    def __init__(self, directory_path, an, mp_id, filter, ref_star_data, mp_data):
        """ Contains all FITS information for one session (one an "Astronight", one minor planet "MP").
        :param directory_path:
        :param an:
        :param mp_id:
        :param filter:

        """
        self.directory = directory_path
        self.an = an
        self.mp_id = mp_id
        self.filter = filter
        all_filenames = get_mp_filenames(directory_path)
        all_images = [WholeImage(directory_path, fn) for fn in all_filenames]
        self.wholeimages = [ai for ai in all_images if ai.filter == self.filter]
        self.ref_star_data = ref_star_data
        self.mp_data = mp_data

    def make_subarray_list(self):
        pass


class WholeImage:
    def __init__(self, directory, filename):
        self.fullpath = os.path.join(directory, filename)
        self.filename = filename
        self.image = CCDData.read(self.fullpath, unit='adu')
        self.filter = self.image.meta['Filter']
        self.exposure = self.image.meta['Exposure']
        self.jd = self.image.meta['JD']
        self.jd_mid = self.jd + (self.exposure / 24 / 3600) / 2
        self.ref_star_radecs = []
        self.mp_radec = None

    def rough_align(self, target_xy):
        pass

    def make_subarray(self, x_first, x_last, y_first, y_last):
        pass


class SubarrayList:
    def __init__(self):
        pass


class Subarray:
    def __init__(self):
        pass
