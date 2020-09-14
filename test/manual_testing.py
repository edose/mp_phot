__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
from math import floor
import pytest

import numpy as np
import astropy.io.fits as fits
from astropy.nddata import CCDData
from astropy.modeling.models import Gaussian2D
import matplotlib.pyplot as plt

from mp_phot.measure import MP_ImageList, plot_arrays
from mp_phot.do_workflow import Settings

from mp_phot.util import get_mp_filenames

MP_ADDED_TEST_FITS_DIR = 'C:/Dev/mp_phot/test/$sessions_for_test/MP_1911/AN20200617'


def test_mp_added_fits():
    mp_file_early = 'MP_191-0001-Clear_Added.fts'
    mp_file_late = 'MP_191-0028-Clear_Added.fts'
    # ref_star_list all in mp_file_early.
    ref_star_list = [{'name': 'Sparse_bkgd',  'xy1': (1228, 600), 'xy2': (1585, 587)},
                     {'name': 'Dense_bkgd',   'xy1': (1892, 821), 'xy2': (2218, 1401)},
                     {'name': 'One brt star', 'xy1': (1461.6, 1370), 'xy2': (1563.4, 1668.7)}]
    mp_list = [{'name': 'Sparse_bkgd',  'xy1_early': (1510, 698),     'xy1_late': (1746, 646)},
               {'name': 'Dense_bkgd',   'xy1_early': (1897.9, 989.1), 'xy1_late': (2233.0, 1141.0)},
               {'name': 'One brt star', 'xy1_early': (1368, 1544),    'xy1_late': (1587, 1533)}]

    for ref, mp in zip(ref_star_list, mp_list):
        ref_star_locations = [(mp_file_early, ref['xy1'][0], ref['xy1'][1]),
                              (mp_file_early, ref['xy2'][0], ref['xy2'][1])]
        mp_location_early = (mp_file_early, mp['xy1_early'][0], mp['xy1_early'][1])
        mp_location_late = (mp_file_late, mp['xy1_late'][0], mp['xy1_late'][1])
        mp_locations = [mp_location_early, mp_location_late]
        imlist = MP_ImageList.from_fits(MP_ADDED_TEST_FITS_DIR, '191', '20200617', 'Clear',
                                        ref_star_locations=ref_star_locations, mp_locations=mp_locations,
                                        settings=Settings())
        imlist.calc_ref_star_radecs()
        imlist.calc_mp_radecs()
        imlist.make_subimages()
        imlist.get_subimage_locations()
        subarray_list = imlist.make_subarrays()
        subarray_list.make_matching_kernels()
        subarray_list.convolve_subarrays()
        subarray_list.realign()
        subarray_list.make_best_bkgd_array()
        subarray_list.make_mp_only_subarrays()
        # plot_arrays('MP-only (bkgd-subtr) subarrays',
        #                     [sa.realigned_mp_only_array for sa in subarray_list.subarrays],
        #                     [sa.filename for sa in subarray_list.subarrays])
        subarray_list.do_mp_aperture_photometry()
        # df_mp_only = subarray_list.make_df_mp_only()
        print('\nCase:', mp['name'])
        for i, sa in enumerate(subarray_list.subarrays):
            print('   ', sa.filename, '   ',
                  'MP flux=', '{:.3f}'.format(sa.mp_flux), '   ',
                  'flux sigma=', '{:.3f}'.format(sa.mp_sigma))
        # plt.show()


def do_raw_aperture_photometry(imlist):
    """ Do aperture photometry on original FITS images (for comparison with improved algorithms).
    :param imlist: MP_ImageList *after* .calc_mp_radecs() has been run. [MP_ImageList object]
    :return: [None] output is printed to console.
    """
    from mp_phot.util import Square
    for mpi in imlist.mp_images:
        this_masked_array = np.ma.array(data=mpi.image.data.copy(),
                                        mask=mpi.image.mask.copy())
        mp_square = Square(this_masked_array,
                                )






# def add_test_mps(read_directory_path=READ_TEST_FITS_DIR, write_directory_path=WRITE_TEST_FITS_DIR,
#                  this_filter='Clear', flux=100000, sigma=5):
#     """ Add constant-flux, constant-shape, moving MP-like sources to all FITS files in a path.
#         To each filename, add '_Added', so e.g., 'MP_191-0001-Clear_Added.fts'.
#         Usage: util.add_test_mps() for default MP FITS written to new directory.
#     :param read_directory_path:
#     :param write_directory_path:
#     :param this_filter:
#     :param flux:
#     :param sigma:
#     :return:
#     """
#     # Define locations and shape of MP signals to add:
#     mp_file_early = 'MP_191-0001-Clear.fts'
#     mp_file_late = 'MP_191-0028-Clear.fts'
#     mps_to_add = [{'name': 'Sparse_bkgd',  'xy1_early': (1510, 698),  'xy1_late': (1746, 646)},
#                {'name': 'Dense_bkgd',   'xy1_early': (1897.9, 989.1),  'xy1_late': (2233.0, 1141.0)},
#                {'name': 'One brt star', 'xy1_early': (1368, 1544), 'xy1_late': (1587, 1533)}]
#     half_size = int(floor(4 * sigma))
#     edge_length = 2 * half_size + 1
#     y, x = np.mgrid[0:edge_length, 0:edge_length]
#     gaussian = Gaussian2D(1, half_size, half_size, sigma, sigma)
#     source_psf = gaussian(x, y)
#     source_psf *= (flux / np.sum(source_psf))  # set total area to desired flux.
#
#     # Generate arrays containing new MP fluxes:
#     for i, mp in enumerate(mps_to_add):
#         mp_location_early = (mp_file_early, mp['xy1_early'][0], mp['xy1_early'][1])
#         mp_location_late = (mp_file_late, mp['xy1_late'][0], mp['xy1_late'][1])
#         mp_locations = [mp_location_early, mp_location_late]
#         imlist = MP_ImageList.from_fits(READ_TEST_FITS_DIR, '191', '20200617', 'Clear',
#                                         ref_star_locations=[],  # ref_star not needed here.
#                                         mp_locations=mp_locations, settings=Settings())
#         if i == 0:
#             # arrays will accumulate new MP signals; using a dict to ensure sync on writing.
#             output_dict = {mpi.filename: mpi.array.copy() for mpi in imlist.mp_images}
#         imlist.calc_mp_radecs()
#         for i_mpi, mpi in enumerate(imlist.mp_images):  # i.e., CCDData objects.
#             fn = mpi.filename
#             x0, y0 =




    #
    #
    # # To each image, add requested MP-like sources:
    # # df = start_df_bd(read_directory_path, this_filter)
    # mp_filenames = get_mp_filenames(read_directory_path)
    # mp_ur_filenames = [fn for fn in mp_filenames if '_added' not in fn]
    # # hdu_list = [fits.open(os.path.join(read_directory_path, fn))[0] for fn in mp_ur_filenames]
    #
    # # Read FITS file, add MP signals at correct location, write modified FITS file:
    # fullpaths = [os.path.join(read_directory_path, fn) for fn in mp_ur_filenames]
    # hdu_list = [fits.open(fp)[0] for fp in fullpaths]
    # ccddata_list = [CCDData.read(fp, unit="adu") for fp in fullpaths]
    # jd_start_list = [float(hdu.header['JD']) for hdu in hdu_list]
    # exposure_list = [float(hdu.header['EXPOSURE']) for hdu in hdu_list]
    # jd_mid_list = [jd_start + exp / 3600 / 24 / 2 for (jd_start, exp) in zip(jd_start_list, exposure_list)]
    # jd_span = jd_mid_list[-1] - jd_mid_list[0]
    # ra_dec_list_early = [ccddata_list[0].wcs.all  for source in mps_to_add]
    #
    #
    # for i, fn in enumerate(mp_ur_filenames):
    #     jd_fract = (jd_mid_list[i] - jd_mid_list[0]) / jd_span
    #
    #
    #
    #
    #     for source in mps_to_add:
    #         x0, y0 = tuple(ccddata.wcs.all_world2pix(list(source['xy1_early']), 0, ra_dec_order=True)[0])
    #
    #
    #
    #
    #
    #
    #
    #
    #     df = calc_mp_radecs(df, mp_file_early, mp_file_late, source['xy1_early'], source['xy1_late'])
    #     for i, ccddata in enumerate(df['Image']):
    #         x0, y0 = tuple(ccddata.wcs.all_world2pix([[df.iloc[i]['MP_RA'],
    #                                                    df.iloc[i]['MP_Dec']]], 0,
    #                                                  ra_dec_order=True)[0])
    #         x_base = int(floor(x0)) - half_size  # position in image of PSF's (0,0) origin.
    #         y_base = int(floor(y0)) - half_size  # "
    #         x_psf_center = x0 - x_base  # offset from PSF's origin of Gaussian's center.
    #         y_psf_center = y0 - y_base  # offset from PSF's origin of Gaussian's center.
    #         gaussian = Gaussian2D(1, x_psf_center, y_psf_center, sigma, sigma)
    #         source_psf = gaussian(x, y)
    #         source_psf *= (flux / np.sum(source_psf))  # set total area to desired flux.
    #         source_psf_uint16 = np.round(source_psf).astype(np.uint16)
    #         hdu_list[i].data[y_base:y_base + edge_length,
    #                          x_base:x_base + edge_length] += source_psf_uint16  # np ndarray: [y,x].
    #
    # # Save updated image data to otherwise identical FITS files with new names:
    # for i, filename in enumerate(df['Filename']):
    #     fn, ext = os.path.splitext(filename)
    #     fn_new = fn + '_Added' + ext  # e.g., 'MP_191-0001-Clear_Added.fts'
    #     hdu_list[i].writeto(os.path.join(WRITE_TEST_FITS_DIR, fn_new))