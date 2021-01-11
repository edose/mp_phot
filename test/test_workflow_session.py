__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python native packages:
import os
import shutil

# External packages:
import pytest

# From target package:
import mp_phot.workflow_session as ws
import mp_phot.ini as ini
from mp_phot.util import get_mp_filenames

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SESSION_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'test', '$sessions_for_test')
TEST_MP = 191
TEST_AN = 20200617
NEW_TEST_MP = 1111


_____TEST_UTILITY_functions____________________________ = 0


def test_make_control_dict():
    mp_dir_string = 'MP_' + str(TEST_MP)
    an_string = 'AN' + str(TEST_AN)
    dir_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, mp_dir_string, an_string)
    os.chdir(dir_path)  # start at source directory.
    cd = ws.make_control_dict()
    assert isinstance(cd, dict)
    assert len(cd['ref star xy']) == 3
    assert cd['ref star xy'][1] == ('MP_191-0001-Clear.fts', 819.3, 1011.7)
    assert len(cd['mp xy']) == 2
    assert cd['mp xy'][0] == ('MP_191-0001-Clear.fts', 826.4, 1077.4)
    assert cd['mp xy'][1] == ('MP_191-0028-Clear.fts', 1144.3, 1099.3)
    assert set(cd['omit comps']) == set(['444', '333', '23', '1'])
    assert cd['omit obs'] == []
    assert set(cd['omit images']) == set(['xxx.fts', 'yyy.fts'])
    assert cd['min catalog r mag'] == 10.0
    assert cd['max catalog r mag'] == 16.0
    assert cd['max catalog dr mmag'] == 20.0
    assert cd['min catalog ri color'] == 0.0
    assert cd['max catalog ri color'] == 0.4
    assert cd['mp ri color'] == 0.220
    assert cd['fit transform'] == ('use', '+0.4', '-0.6')
    assert cd['fit extinction'] == 'yes'
    assert cd['fit vignette'] == True
    assert cd['fit xy'] == False
    assert cd['fit jd'] == True


def test_get_context():
    mp_dir_string = 'MP_' + str(TEST_MP)
    an_string = 'AN' + str(TEST_AN)
    dir_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, mp_dir_string, an_string)
    os.chdir(dir_path)  # start at source directory.
    # Normal case:
    this_directory, mp_string, an_string, filter_string = ws.get_context()
    assert this_directory == dir_path
    assert mp_string == '191'
    assert an_string == '20200617'
    assert filter_string == 'Clear'
    # Error case:
    mp_dir_string = 'MP_1911'
    an_string = 'AN' + str(TEST_AN)
    error_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, mp_dir_string, an_string)
    os.chdir(error_path)
    context = ws.get_context()
    assert context is None


def test_orient_this_function():
    mp_dir_string = 'MP_' + str(TEST_MP)
    an_string = 'AN' + str(TEST_AN)
    dir_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, mp_dir_string, an_string)
    os.chdir(dir_path)  # start at source directory.
    result = ws.orient_this_function('calling function name for test')
    assert result is not None
    context, defaults_dict, control_dict, log_file = result
    assert context == ws.get_context()
    assert isinstance(defaults_dict, dict)
    assert defaults_dict.get('session top directory', None) is not None
    assert isinstance(control_dict, dict)
    assert control_dict['mp ri color'] == +0.220



_____TEST_WORKFLOW_functions_________________________ = 0


def test_start():
    """ We need to: create new test directory, fill it with images, run start(), do asserts,
            tear down new test directory. """
    # TODO: Adapt this for workflow_session.py (from do_workflow.py).
    # Set up:
    mp_dir_string = 'MP_' + str(TEST_MP)
    an_string = 'AN' + str(TEST_AN)
    print('\n============ TEST START ============')  # (first test always needs a linefeed)
    new_dir_path = make_test_session_directory(TEST_SESSION_TOP_DIRECTORY, TEST_MP, NEW_TEST_MP, TEST_AN)
    source_dir_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, mp_dir_string, an_string)
    os.chdir(source_dir_path)  # start at source directory.
    defaults_dict = ini.make_defaults_dict()
    log_filename = defaults_dict['session log filename']
    new_log_fullpath = os.path.join(new_dir_path, log_filename)

    # start() must go to new directory and make proper log file:
    assert not os.path.isfile(new_log_fullpath)  # before start().
    assert os.getcwd() == source_dir_path          # "
    ws.start(TEST_SESSION_TOP_DIRECTORY, NEW_TEST_MP, TEST_AN, 'Clear')
    assert os.getcwd() == new_dir_path
    assert os.path.isfile(new_log_fullpath)
    assert set(get_mp_filenames(new_dir_path)) == set(get_mp_filenames(source_dir_path))


def test_resume():
    # test_start() must pass before running this.
    # Set up:
    new_dir_path = make_test_session_directory(TEST_SESSION_TOP_DIRECTORY, TEST_MP, NEW_TEST_MP, TEST_AN)
    ws.start(TEST_SESSION_TOP_DIRECTORY, NEW_TEST_MP, TEST_AN, filter='Clear')
    source_dir_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, 'MP_' + str(TEST_MP), 'AN' + str(TEST_AN))
    os.chdir(source_dir_path)  # start at source directory.

    # resume() must go to new directory and get context from log file:
    assert os.getcwd() == source_dir_path
    ws.resume(TEST_SESSION_TOP_DIRECTORY, NEW_TEST_MP, TEST_AN, filter='Clear')
    assert os.getcwd() == new_dir_path
    assert ws.get_context() == (new_dir_path, str(NEW_TEST_MP), str(TEST_AN), 'Clear')


def test_write_control_ini_stub():
    # Set up:
    new_dir_path = make_test_session_directory(TEST_SESSION_TOP_DIRECTORY, TEST_MP, NEW_TEST_MP, TEST_AN)
    filenames_temporal_order = ['Earliest file.fts', 'Other file.fts', 'Latest file.fts']
    ws.start(TEST_SESSION_TOP_DIRECTORY, NEW_TEST_MP, TEST_AN, filter='Clear')
    defaults_dict = ini.make_defaults_dict()
    control_ini_filename = defaults_dict['session control filename']
    fullpath = os.path.join(new_dir_path, control_ini_filename)

    # Write new control.ini file & test:
    assert not os.path.exists(fullpath)
    ws.write_control_ini_stub(new_dir_path, filenames_temporal_order)
    assert os.path.exists(fullpath)
    with open(fullpath, 'r') as f:
        lines = f.readlines()
    assert lines[0].startswith('# This is ' + fullpath)
    assert any([filenames_temporal_order[0] in line for line in lines])
    assert any([filenames_temporal_order[-1] in line for line in lines])
    assert not any([filenames_temporal_order[1] in line for line in lines])

    # Must not overwrite existing control.ini file:
    new_filenames_temporal_order = ['New Earliest file.fts', 'New Other file.fts', 'New Latest file.fts']
    assert os.path.exists(fullpath)
    ws.write_control_ini_stub(new_dir_path, new_filenames_temporal_order)
    assert os.path.exists(fullpath)
    with open(fullpath, 'r') as f:
        lines = f.readlines()
    assert any([filenames_temporal_order[0] in line for line in lines])      # old content remains.
    assert any([filenames_temporal_order[-1] in line for line in lines])     # "
    assert not any([filenames_temporal_order[1] in line for line in lines])  # "
    assert not any([new_filenames_temporal_order[0] in line for line in lines])   # not overwritten.
    assert not any([new_filenames_temporal_order[-1] in line for line in lines])  # not overwritten.
    assert not any([new_filenames_temporal_order[1] in line for line in lines])   # not overwritten.

    # Ensure that a proper control_dict can be made from this control.ini file:
    cd = ws.make_control_dict()  # this fn previously tested, see above.
    assert len(cd['ref star xy']) == 3
    assert cd['ref star xy'][2][0] == filenames_temporal_order[0]
    assert len(cd['mp xy']) == 2
    assert cd['mp xy'][filenames_temporal_order[0]] == (0.0, 0.0)
    assert cd['omit comps'] == cd['omit obs'] == cd['omit images'] == []
    assert cd['fit jd'] == True


def test_assess():
    """ Test dict of warnings and errors (visually inspect those currently printed). """
    # Set up:
    new_dir_path = make_test_session_directory(TEST_SESSION_TOP_DIRECTORY, TEST_MP, NEW_TEST_MP, TEST_AN)
    ws.start(TEST_SESSION_TOP_DIRECTORY, NEW_TEST_MP, TEST_AN, filter='Clear')
    d = ws.assess(return_results=True)
    assert d['file not read'] == d['filter not read'] == []
    assert set(d['file count by filter']) == set([('Clear', 5), ('R', 1), ('I', 1)])
    assert d['warning count'] == 0
    assert d['not platesolved'] == d['not calibrated'] == []
    assert d['unusual fwhm'] == d['unusual focal length'] == []
    defaults_dict = ini.make_defaults_dict()
    log_filename = defaults_dict['session log filename']
    log_file_fullpath = os.path.join(new_dir_path, log_filename)
    assert os.path.isfile(log_file_fullpath)
    control_filename = defaults_dict['session control filename']
    control_fullpath = os.path.join(new_dir_path, control_filename)
    assert os.path.isfile(control_fullpath)


def test_make_df_images():
    # Use MP_191 as is.
    ws.resume(TEST_SESSION_TOP_DIRECTORY, TEST_MP, TEST_AN, filter='Clear')
    df_images = ws.make_df_images()
    assert len(df_images) == 7
    assert len(list(df_images.columns)) == 9
    assert list(df_images.columns[:5]) == ['FITSfile', 'JD_mid', 'Filter', 'Exposure', 'Airmass']
    assert set(df_images.columns[5:]) == set(['JD_start', 'UTC_start', 'UTC_mid', 'JD_fract'])
    assert list(df_images.index) == list(df_images.FITSfile)
    assert 'MP_191-0006-Clear.fts' in df_images['FITSfile']


_____HELPER_FUNCTIONS______________________________________ = 0


def make_test_session_directory(session_top_directory, source_test_mp, new_test_mp, test_an):
    """ Make a fresh test directory (probably with dir mp not matching filename mps).
    :param session_top_directory:
    :param source_test_mp:
    :param new_test_mp:
    :param test_an:
    :return:
    """
    import shutil
    # Delete previous test session directory, if exists:
    new_mp_path = os.path.join(session_top_directory, 'MP_' + str(new_test_mp))
    new_dir_path = os.path.join(new_mp_path, 'AN' + str(test_an))
    shutil.rmtree(new_dir_path, ignore_errors=True)  # NB: this doesn't always work in test environment.
    shutil.rmtree(new_mp_path, ignore_errors=True)   # NB: this doesn't always work in test environment.
    # Make new test session directory (FITS only, no other files):
    source_dir_path = os.path.join(session_top_directory, 'MP_' + str(source_test_mp), 'AN' + str(test_an))
    fits_filenames = get_mp_filenames(source_dir_path)
    os.makedirs(new_dir_path, exist_ok=True)
    for fn in fits_filenames:
        source_fullpath = os.path.join(source_dir_path, fn)
        shutil.copy2(source_fullpath, new_dir_path)
    return new_dir_path
