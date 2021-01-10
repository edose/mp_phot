__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
import pytest
import pandas as pd


from mp_phot import workflow_session
from mp_phot import util


MP_PHOT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SESSIONS_DIRECTORY = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'test', '$sessions_for_test')
TEST_MP = '191'
TEST_AN = '20200617'
DEFAULTS_FULLPATH = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'data', 'defaults.txt')
LOG_FILENAME = 'mp_phot.log'
CONTROL_FILENAME = 'control.txt'


SUPPORT_FUNCTIONS____________________________________ = 0


def test_get_context():
    # Target stays in workflow_session.py (workflow_color.py will need its own version of this).
    # Ensure context is None when not in a prepared directory:
    os.chdir(MP_PHOT_ROOT_DIRECTORY)
    reset_test_directory()
    context = workflow_session.get_context()
    assert context is None
    # Ensure context is valid when in a prepared directory:
    target_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617')
    os.chdir(target_directory)
    workflow_session.start(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN, 'ClearForTest')
    context = workflow_session.get_context()
    assert context is not None
    this_directory, mp_string, an_string, filter_string = context
    assert this_directory == target_directory
    assert mp_string == '#191'
    assert an_string == '20200617'
    assert filter_string == 'ClearForTest'


# def make_test_control_txt():
#     """ In Test directory, deletes all but FITS, sets context to directory, makes a test control.txt. """
#     reset_test_directory()
#     workflow_session.start(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN, 'ClearForTestControl')
#     workflow_session.assess()  # to write control.txt stub.
#     this_directory, _, _, _ = workflow_session.get_context()
#     control_fullpath = os.path.join(this_directory, CONTROL_FILENAME)
#     with open(control_fullpath, 'r') as f:
#         lines = f.readlines()
#     new_lines = lines.copy()
#     ref_star_lines_written, mp_lines_written = 0, 0
#     ref_star_xy = [('MP_191-0001-Clear.fts', 790.6, 1115.0),
#                           ('MP_191-0001-Clear.fts', 819.3, 1011.7),
#                           ('MP_191-0001-Clear.fts', 1060.4, 1066.0)]
#     mp_locations = [('MP_191-0001-Clear.fts', 826.4, 1077.4),
#                     ('MP_191-0028-Clear.fts', 1144.3, 1099.3)]
#     for i, line in enumerate(lines):
#         if line.startswith('#REF_STAR_LOCATION'):
#             new_lines[i] = '#REF_STAR_LOCATION  ' + ref_star_xy[ref_star_lines_written][0] +\
#                            '  ' + str(ref_star_xy[ref_star_lines_written][1]) +\
#                            '  ' + str(ref_star_xy[ref_star_lines_written][2]) + '\n'
#             ref_star_lines_written += 1
#         if line.startswith('#MP_LOCATION'):
#             new_lines[i] = '#MP_LOCATION  ' + mp_locations[mp_lines_written][0] +\
#                            '  ' + str(mp_locations[mp_lines_written][1]) +\
#                            '  ' + str(mp_locations[mp_lines_written][2]) + '\n'
#             mp_lines_written += 1
#     with open(control_fullpath, 'w') as f:
#         f.writelines(new_lines)


WORKFLOW_FUNCTIONS_________________________________________ = 0


# def test_start():
#     print('\n============ TEST START ============')  # (first test always needs a linefeed)
#     os.chdir(MP_PHOT_ROOT_DIRECTORY)
#     reset_test_directory()
#     workflow_session.start(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN, 'ClearForTest')
#     this_directory, mp_string, an_string, filter_string = workflow_session.get_context()
#     assert os.getcwd() == this_directory  # current directory is now test directory.
#     assert mp_string == '#' + TEST_MP
#     assert an_string == TEST_AN
#     assert filter_string == 'ClearForTest'
#     assert os.path.isfile(LOG_FILENAME)
#
#
# def test_resume():
#     os.chdir(MP_PHOT_ROOT_DIRECTORY)
#     reset_test_directory()
#     if os.path.isfile(os.path.join(TEST_SESSIONS_DIRECTORY, LOG_FILENAME)):
#         os.remove(os.path.join(TEST_SESSIONS_DIRECTORY, LOG_FILENAME))
#     workflow_session.start(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN, 'ClearForTestResume')
#     os.chdir(MP_PHOT_ROOT_DIRECTORY)
#     workflow_session.resume(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN, 'ClearForTestResume')
#     this_directory, mp_string, an_string, filter_string = workflow_session.get_context()
#     assert os.getcwd() == this_directory  # current directory is now test directory.
#     assert mp_string == '#' + TEST_MP
#     assert an_string == TEST_AN
#     assert filter_string == 'ClearForTestResume'
#     assert os.path.isfile(LOG_FILENAME)


# def test_class_settings():
#     s = workflow_session.Settings(instrument_name='$test_instrument')
#     # Test some default values:
#     assert s['INSTRUMENT'] == '$test_instrument'  # default doesn't matter because we specified name.
#     assert s['MIN_CATALOG_R_MAG'] != '10'
#     assert s['MIN_CATALOG_R_MAG'] == 10
#     assert s['FIT_TRANSFORM'] == ['Clear', 'SR', 'SR-SI', 'Use', '+0.36', '-0.54']
#     # Test some instrument values:
#     assert s['PIXEL_SHIFT_TOLERANCE'] == 200
#     assert s['FWHM_NOMINAL'] == 5
#     assert s['ADU_SATURATION'] == 54000
#     assert s['PINPOINT_PIXEL_SCALE_FACTOR'] == pytest.approx(0.997)
#     assert s['DEFAULT_FILTER'] == 'Clear'
#     assert s['TRANSFORM'][0] == ['Clear', 'SR', 'SR-SI', 'Use', '+0.36', '-0.54']
#     assert s['TRANSFORM'][1] == ['BB',    'SR', 'SR-SI', 'Fit=2']
#
#     def test_assess():
#         """ Little testing here; most testing happens by running and seeing console output. """
#         workflow_session.resume(TEST_SESSIONS_DIRECTORY, TEST_MP, TEST_AN)
#         workflow_session.assess()
#         this_directory, mp_string, an_string, _ = workflow_session.get_context()
#         assert os.getcwd() == this_directory  # current directory remains test directory.
#         assert mp_string == '#' + TEST_MP
#         assert an_string == TEST_AN
#         assert os.path.isfile(LOG_FILENAME)
#         assert os.path.isfile(CONTROL_FILENAME)
#
#     # ref_star_xy = [('MP_191-0001-Clear.fts', 1000, 100), ('MP_191-0001-Clear.fts', 1500, 1000)]
#     # mp_locations = [('MP_191-0001-Clear.fts', 100, 1000), ('MP_191-0028-Clear.fts', 200, 1050)]


# def test_class_control():
#     make_test_control_txt()
#     c = workflow_session.Control()
#     assert c['IS_VALID'] is True
#     assert c['ERRORS'] == []
#     assert c['MAX_CATALOG_DR_MMAG'] == 20.0
#     assert c['FIT_VIGNETTE'] is True
#     assert len(c['REF_STAR_LOCATION']) == 3
#     assert c['REF_STAR_LOCATION'][0] == ['MP_191-0001-Clear.fts', 790.6, 1115.0]
#     assert c['REF_STAR_LOCATION'][1] == ['MP_191-0001-Clear.fts', 819.3, 1011.7]
#     assert len(c['MP_LOCATION']) == 2
#     assert c['MP_LOCATION'][1] == ['MP_191-0028-Clear.fts', 1144.3, 1099.3]
#     reset_test_directory()  # to delete now-contaminated control.txt.


HELPER_FUNCTIONS_________________________________________ = 0


def reset_test_directory():
    """ Simply removes MP files from test directory. """
    test_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    all_filenames = pd.Series([e.name for e in os.scandir(test_directory) if e.is_file()])
    for fn in all_filenames:
        fullpath = os.path.join(test_directory, fn)
        if not (('.' + fn.rsplit('.', maxsplit=1)[1]) in util.VALID_FITS_FILE_EXTENSIONS):
            os.remove(fullpath)
    all_filenames = pd.Series([e.name for e in os.scandir(test_directory) if e.is_file()])
