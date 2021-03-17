__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import pytest
import numpy as np
import pandas as pd
from astropy.nddata import CCDData
import astropy.io.fits as apyfits

# From this package:
from mp_phot import util
# from mp_phot import workflow_session

MP_PHOT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SESSIONS_DIRECTORY = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'test', '$sessions_for_test')
TEST_MP = '191'
TEST_AN = '20200617'
MP_AN_DIRECTORY = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
DEFAULTS_FULLPATH = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'data', 'defaults.txt')


def test_get_mp_filenames():
    this_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    mp_filenames = util.get_mp_filenames(this_directory)
    assert isinstance(mp_filenames, list)
    assert all([isinstance(fn, str) for fn in mp_filenames])
    assert len(mp_filenames) == 7
    assert all([fn.startswith('MP_') for fn in mp_filenames])
    assert all([fn[-4:] in util.VALID_FITS_FILE_EXTENSIONS for fn in mp_filenames])
    assert len(set(mp_filenames)) == len(mp_filenames)  # filenames are unique.


def test_get_mp_and_an_strings():
    # Normal cases:
    mp_str, an_str = util.get_mp_and_an_strings(1108, 20200617)  # case: integers.
    assert mp_str == '1108'
    assert an_str == '20200617'
    mp_id, an_string = util.get_mp_and_an_strings('1108', '20200617')  # case: strs (MP numbered).
    assert mp_id == '1108'
    assert an_string == '20200617'
    mp_id, an_string = util.get_mp_and_an_strings('1997 TX3', '20200617')  # case: strs (MP unnumbered).
    assert mp_id == '~1997 TX3'
    assert an_string == '20200617'
    # Error cases:
    return_value = util.get_mp_and_an_strings(1108, 22200617)  # bad AN date.
    assert return_value is None
    return_value = util.get_mp_and_an_strings(1108, 'hahaha')  # AN date doesn't represent int.
    assert return_value is None
    return_value = util.get_mp_and_an_strings(1108, 20200617.5)  # AN not neither int nor string.
    assert return_value is None


def test_fits_header_value():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    assert util.fits_header_value(hdu, 'FILTER') == 'Clear'
    assert util.fits_header_value(hdu, 'NAXIS1') == 3072
    assert util.fits_header_value(hdu, 'INVALID') is None


def test_fits_is_plate_solved():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    assert util.fits_is_plate_solved(hdu) is True
    hdu.header['CRVAL1'] = 'INVALID'
    assert util.fits_is_plate_solved(hdu) is False
    hdu = apyfits.open(fullpath)[0]
    del hdu.header['CD2_1']
    assert util.fits_is_plate_solved(hdu) is False


def test_fits_is_calibrated():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    assert util.fits_is_calibrated(hdu) is True
    hdu.header['CALSTAT'] = 'INVALID'
    assert util.fits_is_calibrated(hdu) is False
    del hdu.header['CALSTAT']
    assert util.fits_is_calibrated(hdu) is False


def test_fits_focal_length():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    focallen_value = util.fits_focal_length(hdu)
    assert focallen_value == 2713.5
    del hdu.header['FOCALLEN']
    fl_value = util.fits_focal_length(hdu)
    assert 0.97 * focallen_value <= fl_value <= 1.03 * focallen_value


def test_calc_background_adus():
    # Test without mask in target array/slice:
    fullpath = os.path.join(MP_AN_DIRECTORY, 'MP_191-0001-Clear.fts')
    image = CCDData.read(fullpath, unit='adu')[447:528, 273:354]
    bk, std = util.calc_background_adus(image.data, image.mask)
    assert isinstance(bk, float) and isinstance(std, float)
    assert 155 < bk < 157
    assert 12.8 < std < 13.3

    image = CCDData.read(fullpath, unit='adu')[304:345, 679:720]
    bk, std = util.calc_background_adus(image.data, image.mask)
    assert 155 < bk < 167
    assert 12 < std < 13

    # Test with active mask in target array/slice:
    image = CCDData.read(fullpath, unit='adu')[304:345, 679:720]
    image.mask = np.full_like(image, True, dtype=np.bool)
    image.mask[19:21, 19:21] = False  # expose only the center (where a star lies).
    bk, std = util.calc_background_adus(image.data, image.mask)
    assert 2080 < bk < 2090
    assert 125 < std < 130

    image = CCDData.read(fullpath, unit='adu')[304:345, 679:720]
    image.mask = np.full_like(image, True, dtype=np.bool)
    image.mask[19:21, 19:21] = False  # expose only the center (where a star lies).
    bk, std = util.calc_background_adus(image.data, image.mask, invert_mask=True)
    assert 152 < bk < 160
    assert 12 < std < 13


def test_class_square():
    fullpath = os.path.join(MP_AN_DIRECTORY, 'MP_191-0001-Clear.fts')
    image = CCDData.read(fullpath, unit='adu')
    # Constructor (CCDData case):
    sq = util.Square(image, 313, 487, 40, None)
    assert isinstance(sq, util.Square)
    assert sq.x_center == 313
    assert sq.y_center == 487
    assert sq.radius == 40
    assert sq.mask_radius is None
    assert sq.x_low == 313 - 40
    assert sq.x_high == 313 + 40
    assert sq.y_low == 487 - 40
    assert sq.y_high == 487 + 40
    # assert isinstance(sq.image, CCDData)
    assert sq.shape == (81, 81)
    assert all([sq.data[i, i] == image.data[487 + i - 40, 313 + i - 40] for i in range(sq.shape[0])])
    assert sq.is_valid is True
    assert sq.is_cropped is False
    assert isinstance(sq.data, np.ndarray)   # synonym
    # assert isinstance(sq.array, np.ndarray)  # synonym
    assert sq.parent.shape == image.shape
    assert sq.data[30, 40] == image.data[487 + 30, 313 + 40]  # nb: indices are [y,x]
    assert sq.mask.shape == sq.shape
    assert np.any(sq.mask) == False

    # Constructor (numpy ndarray case):
    array = image.data
    sq = util.Square(array, 313, 487, 40, None)
    # assert isinstance(sq.image, np.ndarray)
    assert isinstance(sq.data, np.ndarray)
    assert sq.is_valid is True
    assert sq.mask.shape == sq.shape
    assert np.any(sq.mask) == False

    # Outside range gives invalid object:
    sq = util.Square(image, 10000, -100, 40, None)
    assert sq.is_valid is False

    # Partial image (cropped at edges):
    sq = util.Square(image, 13, 37, 40, None)
    assert sq.x_center == 13
    assert sq.y_center == 37
    assert sq.x_low == 0
    assert sq.x_high == 13 + 40
    assert sq.y_high == 37 + 40
    assert sq.is_valid is True
    assert sq.is_cropped is True
    assert sq.shape == (78, 54)
    assert sq.data[37, 13] == sq.parent.data[37, 13]  # nb ndarrays indexed [y,x]

    # Verify rounding correct:
    sq = util.Square(image, 313.7, 487.2, 40.7, 20)
    assert sq.x_low == int(round(313.7)) - 40     # x was rounded to nearest integer.
    assert sq.y_low == int(round(487.2)) - 40     # y was rounded to nearest integer.
    assert sq.shape == (81, 81)  # radius 40.7 was rounded down.

    # Verify mask behaves properly:
    sq = util.Square(image, 313, 487, 40, 20)
    assert all([sq.mask[41, i] == True for i in range(19)])
    assert all([sq.mask[41, i] == False for i in range(21, 59)])
    assert all([sq.mask[41, i] == True for i in range(61, 81)])
    assert all([sq.mask[i, 41] == True for i in range(19)])
    assert all([sq.mask[i, 41] == False for i in range(21, 59)])
    assert all([sq.mask[i, 41] == True for i in range(61, 81)])

    # Test .centroid():
    sq = util.Square(image, 703, 328, 20, 10)  # centered 4,4 pixels from center of BRIGHT star.
    assert sq.centroid() == sq.centroid(background_region='inverse')  # verify default parm.
    assert sq.centroid(None) == sq.centroid('none')
    c = sq.centroid()  # normal case, default being 'inverse', using inverse of star mask for background.
    assert c[0] == pytest.approx(700.47, abs=0.1)
    assert c[1] == pytest.approx(325.20, abs=0.1)
    c = sq.centroid(background_region='all')  # using entire Square for background.
    assert c[0] == pytest.approx(700.47, abs=0.1)
    assert c[1] == pytest.approx(325.20, abs=0.1)
    c = sq.centroid(background_region='mask')  # using only star mask for background (not normal).
    assert c[0] == pytest.approx(700.29, abs=0.1)
    assert c[1] == pytest.approx(324.99, abs=0.1)
    c = sq.centroid(background_region=None)  # no background subtraction at all.
    assert c[0] == pytest.approx(701.00, abs=0.1)
    assert c[1] == pytest.approx(325.78, abs=0.1)

    sq = util.Square(image, 364, 315, 25, 10)  # centered 4,4 pixels from center of FAINT star.
    c = sq.centroid()
    assert c[0] == pytest.approx(360.46, abs=0.1)
    assert c[1] == pytest.approx(311.78, abs=0.1)
    c = sq.centroid(background_region='all')
    assert c[0] == pytest.approx(360.46, abs=0.1)
    assert c[1] == pytest.approx(311.78, abs=0.1)
    c = sq.centroid(background_region=None)
    assert c[0] == pytest.approx(363.64, abs=0.1)
    assert c[1] == pytest.approx(314.67, abs=0.1)

    # Test .recentroid():
    sq = util.Square(image, 703, 328, 20, 10)  # centered 4,4 pixels from center of BRIGHT star.
    c = sq.recentroid()
    assert c[0] == pytest.approx(699.75, abs=0.1)
    assert c[1] == pytest.approx(324.45, abs=0.1)
    sq = util.Square(image, 364, 315, 25, 10)  # centered 4,4 pixels from center of FAINT star.
    c = sq.recentroid()
    assert c[0] == pytest.approx(359.92, abs=0.1)
    assert c[1] == pytest.approx(311.16, abs=0.1)
    c = sq.recentroid(background_region=None)
    assert c[0] == pytest.approx(363.64, abs=0.1)
    assert c[1] == pytest.approx(314.67, abs=0.1)
    sq = util.Square(image, 364, 315, 25, 3)  # centered 4,4 pixels from center of FAINT star, tiny mask.
    c = sq.recentroid()
    assert c[0] == pytest.approx(362.58, abs=0.1)
    assert c[1] == pytest.approx(313.78, abs=0.1)


def test_make_pill_mask():
    pm = util.make_pill_mask((40, 30), 15, 20, 10, 18, 4)
    assert isinstance(pm, np.ndarray)
    assert pm.shape == (40, 30)  # [y,x]
    assert np.sum(pm == False) == 92  # number of valid pixels.

    pm = util.make_pill_mask((40, 40), 10, 20, 10, 18, 7)
    assert pm.shape == (40, 40)  # [y,x]
    assert np.sum(pm == False) == 179  # number of valid pixels.

    pm = util.make_pill_mask((40, 40), 8, 20, 28, 20, 3)
    assert pm.shape == (40, 40)  # [y,x]
    assert np.sum(pm == False) == 169  # number of valid pixels.


def test_distance_to_line():
    assert util.distance_to_line(3, 4, -1, 7, -1, -33) == 4.0       # horizontal line.
    assert util.distance_to_line(35, 4, -100, 7, 100, 7) == 3.0     # verical line.
    assert util.distance_to_line(15, 12, -12, 43, 13, -17) == 13.0  # normal within line segment.
    assert util.distance_to_line(15, 12, -12, 43, 23, -41) == 13.0  # normal outside line segment.
    assert util.distance_to_line(3, 4, -1, 7, -1, -33, dist_ab=40.0) == 4.0


def test_reorder_df_columns():
    # Construct test df:
    dict_list = []
    dict_list.append({'AA': 1, 'BB': 32, 'CC': 'hahaha'})
    dict_list.append({'AA': 3, 'BB': 41, 'CC': 'hohoho'})
    dict_list.append({'AA': 2, 'BB': -3, 'CC': 'dreeep'})
    df = pd.DataFrame(data=dict_list)
    assert list(df.columns) == ['AA', 'BB', 'CC']
    df1 = util.reorder_df_columns(df, ['CC'])
    assert list(df1.columns) == ['CC', 'AA', 'BB']
    assert (df1.iloc[1])['CC'] == 'hohoho'
    df2 = util.reorder_df_columns(df, ['CC'], ['AA'])
    assert list(df2.columns) == ['CC', 'BB', 'AA']
    assert (df2.iloc[1])['CC'] == 'hohoho'


_____PROBABLY_NOT_USED_________________________ = 0


def test_shift_2d_array():
    xx, yy = np.meshgrid(range(6), range(5))

    # Test all 9 +/0/- combinations for x and y:
    aa = util.shift_2d_array(xx, 0, 0, 777)
    assert isinstance(aa, np.ndarray)
    assert np.array([aa == xx]).all()
    aa = util.shift_2d_array(xx, 0, 3, 777)
    assert list(aa[0, :]) == 6 * [777]
    assert list(aa[3, :]) == [0, 1, 2, 3, 4, 5]
    assert list(aa[:, 3]) == [777, 777, 777, 3, 3]
    aa = util.shift_2d_array(xx, 0, -2, 777)
    assert list(aa[0, :]) == [0, 1, 2, 3, 4, 5]
    assert list(aa[3, :]) == 6 * [777]
    assert list(aa[:, 3]) == [3, 3, 3, 777, 777]
    aa = util.shift_2d_array(xx, 2, 0, 777)
    assert list(aa[0, :]) == [777, 777, 0, 1, 2, 3]
    assert list(aa[3, :]) == [777, 777, 0, 1, 2, 3]
    assert list(aa[:, 3]) == 5 * [1]
    aa = util.shift_2d_array(xx, -3, 0, 777)
    assert list(aa[0, :]) == [3, 4, 5, 777, 777, 777]
    assert list(aa[3, :]) == [3, 4, 5, 777, 777, 777]
    assert list(aa[:, 1]) == 5 * [4]
    assert list(aa[:, 3]) == 5 * [777]
    aa = util.shift_2d_array(xx, 2, 3, -10)
    assert list(aa[0, :]) == 6 * [-10]
    assert list(aa[:, 3]) == 3 * [-10] + 2 * [1]
    aa = util.shift_2d_array(xx, 2, -3, -10)
    assert list(aa[0, :]) == [-10, -10, 0, 1, 2, 3]
    assert list(aa[3, :]) == 6 * [-10]
    assert list(aa[:, 3]) == [1, 1, -10, -10, -10]
    aa = util.shift_2d_array(xx, -1, 3, -10)
    assert list(aa[0, :]) == 6 * [-10]
    assert list(aa[3, :]) == [1, 2, 3, 4, 5, -10]
    assert list(aa[:, 3]) == [-10, -10, -10, 4, 4]
    aa = util.shift_2d_array(xx, -1, -2, -10)
    assert list(aa[0, :]) == [1, 2, 3, 4, 5, -10]
    assert list(aa[3, :]) == 6 * [-10]
    assert list(aa[:, 3]) == [4, 4, 4, -10, -10]

    # Test edge cases:
    bb = util.shift_2d_array(xx, 100, 100, -10)  # test wild shift extents.
    assert np.array([bb == -10]).all()
    bb = util.shift_2d_array(xx, 100, 100, np.nan)
    assert (bb == np.full_like(xx, np.nan)).all()
    assert bb.shape == np.full_like(bb, np.nan).shape
    cc = util.shift_2d_array(xx, 100, 100)   # test default fill value.
    assert (cc == np.full_like(xx, np.nan)).all()
    assert cc.shape == np.full_like(xx, np.nan).shape