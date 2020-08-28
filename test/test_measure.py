__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import pytest
from astropy.nddata import CCDData
import numpy as np

# From this package:
from mp_phot import measure
from mp_phot import do_workflow
from mp_phot import util
from test.test_do_workflow import make_test_control_txt

MP_PHOT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SESSIONS_DIRECTORY = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'test', '$sessions_for_test')
TEST_MP = '191'
TEST_AN = '20200617'


def test_class_mp_image():
    # Test constructor, just picking first test FITS:
    mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    mp_filenames = util.get_mp_filenames(mp_directory)
    fn = mp_filenames[0]
    im = measure.MP_Image(mp_directory, fn, do_workflow.Settings('Borea'))
    assert im.filter == 'Clear'
    assert im.exposure == 67.0
    assert im.jd == pytest.approx(2459018.6697821761)
    assert im.jd_mid == pytest.approx(im.jd + (im.exposure / 2) / 24 / 3600)
    assert im.ref_star_radecs == []
    assert im.mp_radec is None
    assert im.image.shape == (2047, 3072)
    assert measure.PIXEL_FACTOR_HISTORY_TEXT in im.image.meta['HISTORY']


def test_class_mp_imagelist_1():
    # Set up non-image data required:
    mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    make_test_control_txt()
    control_data = do_workflow.Control()
    ref_star_locations = control_data['REF_STAR_LOCATION']
    mp_locations = control_data['MP_LOCATION']
    settings = do_workflow.Settings()

    # Test primary constructor (with pre-made MP_Image objects):
    all_mp_filenames = util.get_mp_filenames(mp_directory)
    all_images = [measure.MP_Image(mp_directory, fn, settings) for fn in all_mp_filenames]
    imlist = measure.MP_ImageList(mp_directory, TEST_MP, TEST_AN, 'Clear', all_images,
                                  ref_star_locations, mp_locations, settings)
    assert imlist.directory == mp_directory
    assert imlist.mp_id == TEST_MP
    assert imlist.an == TEST_AN
    assert imlist.filter == 'Clear'
    assert len(imlist.mp_images) == 5
    assert all([im.filter == 'Clear' for im in imlist.mp_images])  # each image must have correct filter.
    for i in range(1, len(imlist.mp_images) - 1):
        assert imlist.mp_images[i].jd_mid > imlist.mp_images[i - 1].jd_mid  # chronological order.
    assert len(imlist.ref_star_locations) == 3
    assert imlist.ref_star_locations[0] == ['MP_191-0001-Clear.fts', 790.6, 1115.0]
    assert len(imlist.mp_locations) == 2
    assert imlist.mp_locations[1] == ['MP_191-0028-Clear.fts', 1144.3, 1099.3]
    for mp_image in imlist.mp_images:
        fn = mp_image.filename
        assert imlist[fn].filename == fn
        assert imlist[fn].jd_mid == mp_image.jd_mid

    # Test .from_fits() classmethod constructor (making MP_Image objects from FITS files in directory):
    imlist = measure.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                            ref_star_locations, mp_locations, settings)
    assert imlist.directory == mp_directory
    assert imlist.mp_id == TEST_MP
    assert imlist.an == TEST_AN
    assert imlist.filter == 'Clear'
    assert len(imlist.mp_images) == 5
    assert all([isinstance(im, measure.MP_Image) for im in imlist.mp_images])
    all_mp_filenames = util.get_mp_filenames(mp_directory)
    mpil_filenames = [im.filename for im in imlist.mp_images]
    assert set(mpil_filenames).issubset(set(all_mp_filenames))
    assert len(imlist.ref_star_locations) == 3
    assert imlist.ref_star_locations[0] == ['MP_191-0001-Clear.fts', 790.6, 1115.0]
    assert len(imlist.mp_locations) == 2
    assert imlist.mp_locations[1] == ['MP_191-0028-Clear.fts', 1144.3, 1099.3]


def test_class_mp_imagelist_2():
    # Set up non-image data required:
    mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    make_test_control_txt()
    control_data = do_workflow.Control()
    ref_star_locations = control_data['REF_STAR_LOCATION']
    mp_locations = control_data['MP_LOCATION']
    settings = do_workflow.Settings()

    # Test .calc_ref_star_radecs():
    imlist = measure.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                            ref_star_locations, mp_locations, settings)
    assert imlist.ref_star_radecs == []
    imlist.calc_ref_star_radecs()
    assert len(imlist.ref_star_radecs) == len(imlist.ref_star_locations)
    assert imlist.ref_star_radecs[0][0] == pytest.approx(267.673, 0.001)
    assert imlist.ref_star_radecs[1][1] == pytest.approx(-6.962, 0.001)
    assert imlist.ref_star_radecs[2][0] == pytest.approx(267.622, 0.001)

    # Test .calc_mp_radecs():
    imlist = measure.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                            ref_star_locations, mp_locations, settings)
    assert imlist.mp_radecs == []
    assert imlist.mp_locations_all == []
    imlist.calc_mp_radecs()
    assert len(imlist.mp_radecs) == len(imlist.mp_images)
    assert len(imlist.mp_locations_all) == len(imlist.mp_images)
    filename_0 = imlist.mp_locations[0][0]
    i_0 = imlist.filenames.index(filename_0)
    mp_radec_0 = imlist.mp_locations_all[i_0]
    assert imlist.mp_locations_all[0] == (pytest.approx(826.16, 0.5), pytest.approx(1077.5, 0.5))
    assert imlist.mp_locations_all[3] == (pytest.approx(1099.83, 0.5), pytest.approx(1090.55, 0.5))

    # Test .make_subimages():
    imlist = measure.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                            ref_star_locations, mp_locations, settings)
    imlist.calc_ref_star_radecs()
    imlist.calc_mp_radecs()
    imlist.make_subimages()
    # measure.plot_subimages('TEST new subimages', imlist)
    assert len(imlist.subimages) == 5
    assert all([isinstance(si, CCDData) for si in imlist.subimages])
    assert all([si.shape == imlist.subimages[0].shape for si in imlist.subimages])

    # Test .wcs_align_subimages() and .trim_nans_from_subimages() together:
    # print('===== Subimages before wcs alignment and trim nans:')
    # radec_top_left = [tuple(si.wcs.all_pix2world([list((0, 0))], 0)[0]) for si in imlist.subimages]
    # for i, radec in enumerate(radec_top_left):
    #     print('  top left   ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))
    # y_br, x_br = imlist.subimages[0].shape
    # radec_bottom_rt = [tuple(si.wcs.all_pix2world([list((x_br, y_br))], 0)[0]) for si in imlist.subimages]
    # for i, radec in enumerate(radec_bottom_rt):
    #     print('  bottom rt  ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))
    #
    # imlist.wcs_align_subimages()
    # imlist.trim_nans_from_subimages()
    # print('===== Subimages after wcs alignment and trim nans:')
    # measure.plot_subimages('TEST aligned & nan-trimmed subimages', imlist)  # ##############
    # radec_top_left = [tuple(si.wcs.all_pix2world([list((0, 0))], 0)[0]) for si in imlist.subimages]
    # for i, radec in enumerate(radec_top_left):
    #     print('  top left   ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))
    # y_br, x_br = imlist.subimages[0].shape
    # radec_bottom_rt = [tuple(si.wcs.all_pix2world([list((x_br, y_br))], 0)[0]) for si in imlist.subimages]
    # for i, radec in enumerate(radec_bottom_rt):
    #     print('  bottom rt  ', str(i), '   {:.6f}'.format(radec[0]), '   {:.6f}'.format(radec[1]))

    # Test .get_subimage_locations():
    imlist.get_subimage_locations()
    assert len(imlist.subimage_mp_locations) == 5

    # Test .make_subarrays():
    subarray_list = imlist.make_subarrays()
    assert isinstance(subarray_list, measure.SubarrayList)
    assert len(subarray_list.subarrays) == 5
    assert all([isinstance(sa.array, np.ndarray) for sa in subarray_list.subarrays])
    assert all([isinstance(sa.mask,  np.ndarray) for sa in subarray_list.subarrays])
    assert all([isinstance(sa.ref_star_locations, list) for sa in subarray_list.subarrays])
    assert isinstance(subarray_list.subarrays[0].ref_star_locations, list)
    assert isinstance(subarray_list.subarrays[0].ref_star_locations[0], tuple)
    assert isinstance(subarray_list.subarrays[0].ref_star_locations[0][0], float)
    assert all([isinstance(sa.mp_location, tuple) for sa in subarray_list.subarrays])
    assert isinstance(subarray_list.subarrays[0].mp_location[0], float)


def test_class_subarraylist():
    # Set up everything up to spawning of subarrays:
    mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    make_test_control_txt()
    control_data = do_workflow.Control()
    ref_star_locations = control_data['REF_STAR_LOCATION']
    mp_locations = control_data['MP_LOCATION']
    settings = do_workflow.Settings()
    imlist = measure.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                            ref_star_locations, mp_locations, settings)
    imlist.calc_ref_star_radecs()
    imlist.calc_mp_radecs()
    imlist.make_subimages()
    imlist.get_subimage_locations()
    subarray_list = imlist.make_subarrays()

    # Setup now done, test .make_matching_kernels():
    subarray_list.make_matching_kernels()
    measure.plot_arrays('Matching kernels', [sa.matching_kernel for sa in subarray_list.subarrays],
                        [sa.filename for sa in subarray_list.subarrays])

    subarray_list.convolve_subarrays()
    measure.plot_arrays('After convolution', [sa.convolved_array for sa in subarray_list.subarrays],
                        [sa.filename for sa in subarray_list.subarrays])

    subarray_list.realign()
    for i_sa, sa in enumerate(subarray_list.subarrays):
        for rsl in subarray_list.realigned_ref_star_locations[i_sa]:
            print('Ref star locations(' + str(i_sa) + '): ' + str(rsl))
        print('MP locations' + str(i_sa) + '): ' + str(sa.realigned_ref_star_location[i_sa]))
    measure.plot_arrays('After realignment', [sa.realigned_array for sa in subarray_list.subarrays],
                        [sa.filename for sa in subarray_list.subarrays])


def test_scikit_transform_stuff():
    import skimage.transform as skt
    source_points = np.array([[1, 1], [1, 3], [5, 2]])
    target = np.array([[1, 0.8], [0.9, 2.7], [4.8, 2.1]])
    transform = skt.estimate_transform(ttype='similarity', src=source_points, dst=target)
    realigned_source = transform(source_points)
    print('\n', str(realigned_source))

    realigned_target = transform.inverse(target)
    print('\n\n', str(realigned_target))




