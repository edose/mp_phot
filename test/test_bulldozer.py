__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import pytest
import numpy as np
from astropy.nddata import CCDData
import matplotlib.pyplot as plt

# From this package:
from mp_phot import bulldozer
from mp_phot import workflow_session
from mp_phot import util
# from test.XXX_test_do_workflow import make_test_control_txt

MP_PHOT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SESSIONS_DIRECTORY = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'test', '$sessions_for_test')
TEST_MP = '191'
TEST_AN = '20200617'


def test_class_mp_image():
    # Test constructor, just picking first test FITS:
    mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    mp_filenames = util.get_mp_filenames(mp_directory)
    fn = mp_filenames[0]
    im = bulldozer.MP_Image(mp_directory, fn)
    assert im.filter == 'Clear'
    assert im.exposure == 67.0
    assert im.jd == pytest.approx(2459018.6697821761)
    assert im.jd_mid == pytest.approx(im.jd + (im.exposure / 2) / 24 / 3600)
    assert im.image.shape == (2047, 3072)
    assert bulldozer.PIXEL_FACTOR_HISTORY_TEXT in im.image.meta['HISTORY']


def test_class_mp_imagelist_1():
    """ Test class MP_ImageList from constructors calls through .make_subimages().
        NB: class MP_ImageList applies to color workflow also, though Workflow session data are used here.
    """
    # Set up:
    workflow_session.resume(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN)
    context, defaults_dict, control_dict, log_file = \
        workflow_session.orient_this_function('test_class_mp_imagelist1')
    this_directory, mp_string, an_string, filter_string = context
    control_dict = workflow_session.make_control_dict()
    all_mp_filenames = util.get_mp_filenames(this_directory)

    # Test primary constructor (with pre-made MP_Image objects):
    all_mp_image_objects = [bulldozer.MP_Image(this_directory, fn) for fn in all_mp_filenames]
    imlist = bulldozer.MP_ImageList(this_directory, TEST_MP, TEST_AN, 'Clear',
                                    all_mp_image_objects, control_dict)
    assert imlist.directory == this_directory
    assert imlist.mp_id == TEST_MP
    assert imlist.an == TEST_AN
    assert imlist.filter == 'Clear'
    assert len(imlist.mp_images) == 5
    assert all([im.filter == 'Clear' for im in imlist.mp_images])  # each image must have correct filter.
    for i in range(1, len(imlist.mp_images) - 1):
        assert imlist.mp_images[i].jd_mid > imlist.mp_images[i - 1].jd_mid  # chronological order.
    assert len(imlist.given_ref_star_xy) == 3
    assert imlist.given_ref_star_xy[0] == ('MP_191-0001-Clear.fts', 790.6, 1115.0)
    assert len(imlist.given_mp_xy) == 2
    assert imlist.given_mp_xy[-1] == ('MP_191-0028-Clear.fts', 1144.3, 1099.3)
    for mp_image in imlist.mp_images:
        fn = mp_image.filename
        assert imlist[fn].filename == fn
        assert imlist[fn].jd_mid == mp_image.jd_mid

    # Test .from_fits() classmethod constructor (making MP_Image objects from FITS files in directory):
    imlist = bulldozer.MP_ImageList.from_fits(this_directory, TEST_MP, TEST_AN, 'Clear', control_dict)
    assert imlist.directory == this_directory
    assert imlist.mp_id == TEST_MP
    assert imlist.an == TEST_AN
    assert imlist.filter == 'Clear'
    assert len(imlist.mp_images) == 5
    assert all([isinstance(im, bulldozer.MP_Image) for im in imlist.mp_images])
    all_mp_filenames = util.get_mp_filenames(this_directory)
    mpil_filenames = [im.filename for im in imlist.mp_images]
    assert set(mpil_filenames).issubset(set(all_mp_filenames))
    assert len(imlist.given_ref_star_xy) == 3
    assert imlist.given_ref_star_xy[0] == ('MP_191-0001-Clear.fts', 790.6, 1115.0)
    assert len(imlist.given_mp_xy) == 2
    assert imlist.given_mp_xy[-1] == ('MP_191-0028-Clear.fts', 1144.3, 1099.3)

    # Test .calc_ref_star_radecs():
    imlist = bulldozer.MP_ImageList.from_fits(this_directory, TEST_MP, TEST_AN, 'Clear', control_dict)
    assert imlist.ref_star_radecs == []
    imlist.calc_ref_star_radecs()
    assert len(imlist.ref_star_radecs) == len(imlist.given_ref_star_xy)
    assert imlist.ref_star_radecs[0][0] == pytest.approx(267.673, 0.001)
    assert imlist.ref_star_radecs[1][1] == pytest.approx(-6.962, 0.001)
    assert imlist.ref_star_radecs[2][0] == pytest.approx(267.622, 0.001)

    # Test .calc_mp_radecs_and_xy():
    # Carry imlist in from previous test section.
    assert imlist.mp_radecs == []
    imlist.calc_mp_radecs_and_xy()
    assert len(imlist.mp_radecs) == len(imlist.mp_images)
    assert imlist.mp_radecs[0][0] == pytest.approx(267.666, abs=0.002)
    assert imlist.mp_radecs[3][1] == pytest.approx(-6.977, abs=0.002)
    assert imlist.images_mp_xy[0][0] == pytest.approx(826.4, abs=0.5)
    assert imlist.images_mp_xy[2][1] == pytest.approx(1098.6, abs=0.5)
    assert (imlist.mp_radecs[0][0] - imlist.mp_start_radecs[0][0]) == pytest.approx(-8e-5, abs=1e-5)
    assert (imlist.mp_radecs[3][1] - imlist.mp_start_radecs[3][1]) == pytest.approx(-3.93e-6, abs=1e-6)
    assert (imlist.mp_end_radecs[0][0] - imlist.mp_radecs[0][0]) == pytest.approx(-8e-5, abs=1e-5)
    assert (imlist.mp_end_radecs[2][1] - imlist.mp_radecs[2][1]) == pytest.approx(-3.93e-6, abs=1e-6)
    assert (imlist.images_mp_xy[0][0] - imlist.images_mp_start_xy[0][0]) == pytest.approx(0.43, abs=0.02)
    assert (imlist.images_mp_xy[4][1] - imlist.images_mp_start_xy[4][1]) == pytest.approx(0.03, abs=0.02)
    assert (imlist.images_mp_end_xy[0][0] - imlist.images_mp_xy[0][0]) == pytest.approx(0.43, abs=0.02)
    assert (imlist.images_mp_end_xy[3][1] - imlist.images_mp_xy[3][1]) == pytest.approx(0.03, abs=0.02)

    # Test .calc_make_subimages():
    # Carry imlist in from previous test section.
    assert imlist.subimages == imlist.subimage_sky_adus == []
    imlist.make_subimages(do_plot=True)
    assert len(imlist.subimages) == len(imlist.subimage_sky_adus) == len(imlist.mp_images)
    assert all([isinstance(si, CCDData) for si in imlist.subimages])
    assert all([si.shape == (231, 403) for si in imlist.subimages])
    assert all([si.unit.name == 'adu' for si in imlist.subimages])
    assert all([isinstance(adu, float) for adu in imlist.subimage_sky_adus])
    assert imlist.subimage_sky_adus == [157.0, 157.0, 147.0, 144.0, 143.0]


def test_class_mp_imagelist_2():
    """ Test class MP_ImageList from .wcs_align_subimages through .make_subarrays() (normal workflow's end).
        NB: class MP_ImageList applies to color workflow also, though Workflow session data are used here.
    """
    # Set up, including workflow's first few MP_ImageList function calls:
    workflow_session.resume(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN)
    context, defaults_dict, control_dict, log_file = \
        workflow_session.orient_this_function('test_class_mp_imagelist1')
    this_directory, mp_string, an_string, filter_string = context
    control_dict = workflow_session.make_control_dict()
    all_mp_filenames = util.get_mp_filenames(this_directory)
    imlist = bulldozer.MP_ImageList.from_fits(this_directory, TEST_MP, TEST_AN, 'Clear', control_dict)
    imlist.calc_ref_star_radecs()
    imlist.calc_mp_radecs_and_xy()
    imlist.make_subimages(do_plot=False)

    # Test .wcs_align_subimages():
    assert len(np.unique([si.wcs.wcs.crota for si in imlist.subimages])) == len(imlist.subimages)
    # assert not any([si.wcs.wcs.crota[0] == pytest.approx(imlist.subimages[0].wcs.wcs.crota[0], abs=0.0001)
    #                 for si in imlist.subimages[1:]])
    imlist.wcs_align_subimages()
    assert len(np.unique([si.wcs.wcs.crota for si in imlist.subimages])) == 1
    # assert all([si.wcs.wcs.crota[0] == pytest.approx(imlist.subimages[0].wcs.wcs.crota[0], abs=0.0001)
    #             for si in imlist.subimages[1:]])

    # Test .trim_nans_from_subimages():
    assert all([si.shape == (231, 403) for si in imlist.subimages])
    assert np.sum(np.isnan(imlist.subimages[3].data)) == 0
    assert np.sum(np.isnan(imlist.subimages[4].data)) == 132
    imlist.trim_nans_from_subimages()
    assert all([si.shape == (230, 402) for si in imlist.subimages])
    assert all([np.sum(np.isnan(si.data)) == 0 for si in imlist.subimages])

    # Test .get_subimage_locations():
    # THIS TEST SECTION IN DEVELOPMENT 2021-01-11 pm.
    iiii = 4
    imlist.get_subimage_locations()





# def test_class_mp_imagelist_2():
#     # Set up non-image data required:
#     mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
#     make_test_control_txt()
#     control_data = workflow_session.Control()
#     ref_star_locations = control_data['REF_STAR_LOCATION']
#     mp_locations = control_data['MP_LOCATION']
#     settings = workflow_session.Settings()
#
#     # Test .calc_mp_radecs_and_xy():

#
#     # Test .make_subimages():
#     imlist = bulldozer.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
#                                               ref_star_locations, mp_locations, settings)
#     imlist.calc_ref_star_radecs()
#     imlist.calc_mp_radecs_and_xy()
#     imlist.make_subimages()
#     # measure.plot_subimages('TEST new subimages', imlist)
#     assert len(imlist.subimages) == 5
#     assert all([isinstance(si, CCDData) for si in imlist.subimages])
#     assert all([si.shape == imlist.subimages[0].shape for si in imlist.subimages])
#
#     # Test .wcs_align_subimages() and .trim_nans_from_subimages() together:
#     # print('===== Subimages before wcs alignment and trim nans:')
#     # radec_top_left = [tuple(si.wcs.all_pix2world([list((0, 0))], 0)[0]) for si in imlist.subimages]
#     # for i, radec in enumerate(radec_top_left):
#     #     print('  top left   ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))
#     # y_br, x_br = imlist.subimages[0].shape
#     # radec_bottom_rt = [tuple(si.wcs.all_pix2world([list((x_br, y_br))], 0)[0]) for si in imlist.subimages]
#     # for i, radec in enumerate(radec_bottom_rt):
#     #     print('  bottom rt  ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))
#     #
#     # imlist.wcs_align_subimages()
#     # imlist.trim_nans_from_subimages()
#     # print('===== Subimages after wcs alignment and trim nans:')
#     # measure.plot_subimages('TEST aligned & nan-trimmed subimages', imlist)  # ##############
#     # radec_top_left = [tuple(si.wcs.all_pix2world([list((0, 0))], 0)[0]) for si in imlist.subimages]
#     # for i, radec in enumerate(radec_top_left):
#     #     print('  top left   ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))
#     # y_br, x_br = imlist.subimages[0].shape
#     # radec_bottom_rt = [tuple(si.wcs.all_pix2world([list((x_br, y_br))], 0)[0]) for si in imlist.subimages]
#     # for i, radec in enumerate(radec_bottom_rt):
#     #     print('  bottom rt  ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))
#
#     # Test .get_subimage_locations():
#     imlist.get_subimage_locations()
#     assert len(imlist.subimage_mp_locations) == 5
#
#     # Test .make_subarrays():
#     subarray_list = imlist.make_subarrays()
#     assert isinstance(subarray_list, bulldozer.SubarrayList)
#     assert len(subarray_list.subarrays) == 5
#     assert all([isinstance(sa.array, np.ndarray) for sa in subarray_list.subarrays])
#     assert all([isinstance(sa.mask,  np.ndarray) for sa in subarray_list.subarrays])
#     assert all([isinstance(sa.ref_star_locations, list) for sa in subarray_list.subarrays])
#     assert isinstance(subarray_list.subarrays[0].ref_star_locations, list)
#     assert isinstance(subarray_list.subarrays[0].ref_star_locations[0], tuple)
#     assert isinstance(subarray_list.subarrays[0].ref_star_locations[0][0], float)
#     assert all([isinstance(sa.mp_location, tuple) for sa in subarray_list.subarrays])
#     assert isinstance(subarray_list.subarrays[0].mp_location[0], float)


def test_class_subarraylist():
    # Set up everything up to spawning of subarrays:
    mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    make_test_control_txt()
    control_data = workflow_session.Control()
    # given_ref_star_xy = control_data['REF_STAR_LOCATION']
    # given_ref_star_xy = [['MP_191-0001-Clear.fts',  790.6, 1115.0],
    #                       ['MP_191-0028-Clear.fts', 1583.2, 1192.3]]  # 28: far but bright
    # given_ref_star_xy = [['MP_191-0001-Clear.fts',  790.6, 1115.0],
    #                       ['MP_191-0028-Clear.fts', 1392.7, 1063.4]]  # 28: v.isolated, mid-distance
    ref_star_locations = [['MP_191-0001-Clear.fts',  790.6, 1115.0],
                          ['MP_191-0028-Clear.fts', 1198.5, 1084.4]]  # 28: close but faint
    mp_locations = control_data['MP_LOCATION']
    settings = workflow_session.Settings()
    imlist = bulldozer.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                              ref_star_locations, mp_locations, settings)
    imlist.calc_ref_star_radecs()
    imlist.calc_mp_radecs_and_xy()
    imlist.make_subimages()
    # measure.plot_subimages('Initial SUBIMAGES', imlist)
    imlist.get_subimage_locations()
    subarray_list = imlist.make_subarrays()
    # measure.plot_subarrays('Initial SUBARRAYS', subarray_list)

    # Setup now done, test .make_matching_kernels():
    subarray_list.make_matching_kernels()
    # measure.plot_arrays('Matching kernels', [sa.matching_kernel for sa in subarray_list.subarrays],
    #                     [sa.filename for sa in subarray_list.subarrays])

    subarray_list.convolve_subarrays()
    # measure.plot_arrays('After convolution', [sa.convolved_array for sa in subarray_list.subarrays],
    #                     [sa.filename for sa in subarray_list.subarrays])

    subarray_list.realign()
    # for i_sa, sa in enumerate(subarray_list.subarrays):
    #     for rsl in sa.realigned_ref_star_locations:
    #         print('Ref star locations(' + str(i_sa) + '): ' + str(rsl))
    #     print('MP locations' + str(i_sa) + '): ' + str(sa.realigned_ref_star_locations))
    # measure.plot_arrays('After realignment', [sa.realigned_array for sa in subarray_list.subarrays],
    #                     [sa.filename for sa in subarray_list.subarrays])

    subarray_list.make_best_bkgd_array()
    # measure.plot_one_array("Averaged (MP-free) Subarray", subarray_list.best_bkgd_array)

    subarray_list.make_mp_only_subarrays()
    bulldozer.plot_arrays('MP-only (bkgd-subtr) subarrays',
                          [sa.realigned_mp_only_array for sa in subarray_list.subarrays],
                          [sa.filename for sa in subarray_list.subarrays])

    subarray_list.do_mp_aperture_photometry()
    for i_sa, sa in enumerate(subarray_list.subarrays):
        print(str(i_sa), 'MP flux=', '{:.3f}'.format(sa.mp_flux),
              '   flux sigma=', '{:.3f}'.format(sa.mp_sigma))

    df_mp_only = subarray_list.make_df_mp_only()
    iiii = 4

    # subarray_list.convolve_full_arrays([mpi.image.data for mpi in imlist.mp_images[0:1]])

    plt.show()



# def test_scikit_transform_stuff():
#     import skimage.transform as skt
#     source_points = np.array([[1, 1], [1, 3], [5, 2]])
#     target = np.array([[1, 0.8], [0.9, 2.7], [4.8, 2.1]])
#     transform = skt.estimate_transform(ttype='similarity', src=source_points, dst=target)
#     realigned_source = transform(source_points)
#     print('\n', str(realigned_source))
#
#     realigned_target = transform.inverse(target)
#     print('\n\n', str(realigned_target))


_____HELPER_FUNCTIONS______________________________ = 0


def make_mp_imagelist():
    pass

