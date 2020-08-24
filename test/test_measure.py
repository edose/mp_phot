__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import pytest
from astropy.nddata import CCDData

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

    # Test .make_subimages:
    imlist = measure.MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                            ref_star_locations, mp_locations, settings)
    imlist.calc_ref_star_radecs()
    imlist.calc_mp_radecs()
    imlist.make_subimages()
    measure.plot_subimages('TEST new subimages', imlist)
    assert len(imlist.subimages) == 5

