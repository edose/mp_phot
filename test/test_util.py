__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
import pytest
from mp_phot import util
from mp_phot import do_workflow

MP_PHOT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SESSION_DIRECTORY = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'test', '$sessions_for_test')
TEST_MP = '191'
TEST_AN = '20200617'
DEFAULTS_FULLPATH = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'data', 'defaults.txt')


def test_get_mp_filenames():
    this_directory = os.path.join(TEST_SESSION_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    mp_filenames = util.get_mp_filenames(this_directory)
    assert len(mp_filenames) == 7
    assert all([fn.startswith('MP_') for fn in mp_filenames])
    assert all([fn[-4:] in util.VALID_FITS_FILE_EXTENSIONS for fn in mp_filenames])
    assert len(set(mp_filenames)) == len(mp_filenames)  # filenames are unique.


def test_dict_from_directives_file():
    d = util.dict_from_directives_file(DEFAULTS_FULLPATH)
    all_keys = set(d.keys())
    required_keys = set(do_workflow.REQUIRED_DEFAULT_DIRECTIVES)
    assert len(required_keys - all_keys) == 0  # all required must be present.
    assert d['INSTRUMENT'] == 'Borea'
    assert d['MAX_CATALOG_R_MAG'] == '16'
    assert d['FIT_JD'] == 'Yes'
    assert 'INVALID_KEY' not in all_keys  # absent key returns None.
