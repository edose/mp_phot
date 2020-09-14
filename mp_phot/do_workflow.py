__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from datetime import datetime, timezone
from collections import Counter
from math import sqrt, cos, sin, pi

# External packages:
import numpy as np
import pandas as pd
import astropy.io.fits as apyfits
from astropy.nddata import CCDData
from astropy.wcs import WCS


# From this package:
from .util import dict_from_directives_file, get_mp_filenames
from .measure import MP_ImageList

# TODO: move most of these path and other constants to defaults.txt (or possibly to instrument file):
DEFAULTS_FILE_FULLPATH = 'C:/Dev/mp_phot/data/defaults.txt'   # TODO: make this relative to package path.
INSTRUMENT_FILE_DIRECTORY = 'C:/Dev/mp_phot/data/instrument'  # TODO: make this relative to package path.
MP_TOP_DIRECTORY = 'C:/Astro/MP Photometry/'
LOG_FILENAME = 'mp_phot.log'
CONTROL_FILENAME = 'control.txt'
DF_OBS_ALL_FILENAME = 'df_obs_all.csv'
DF_IMAGES_ALL_FILENAME = 'df_images_all.csv'
DF_COMPS_ALL_FILENAME = 'df_comps_all.csv'



RADIANS_PER_DEGREE = pi / 180.0
PIXEL_FACTOR_HISTORY_TEXT = 'Pixel scale factor applied by apply_pixel_scale_factor().'

# PLATESOLUTION_KEYS_TO_REMOVE = ['CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
#                                 'ZMAG', 'EPOCH', 'PA', 'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2',
#                                 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'TR1_*', 'TR2_*', 'PLTSOLVD']
REQUIRED_DEFAULT_DIRECTIVES = ['INSTRUMENT', 'MP_RI_COLOR', 'MIN_CATALOG_R_MAG', 'MAX_CATALOG_R_MAG',
                               'MAX_CATALOG_DR_MMAG', 'MIN_CATALOG_RI_COLOR', 'MAX_CATALOG_RI_COLOR',
                               'FIT_TRANSFORM', 'FIT_EXTINCTION', 'FIT_VIGNETTE',
                               'FIT_XY', 'FIT_JD']
REQUIRED_INSTRUMENT_DIRECTIVES = ['PIXEL_SHIFT_TOLERANCE', 'FWHM_NOMINAL', 'CCD_GAIN', 'ADU_SATURATION',
                                  'TRANSFORM']
REQUIRED_CONTROL_DIRECTIVES = ['REF_STAR_LOCATION', 'MP_LOCATION', 'MP_RI_COLOR',
                               'MIN_CATALOG_R_MAG', 'MAX_CATALOG_R_MAG',
                               'MAX_CATALOG_DR_MMAG', 'MIN_CATALOG_RI_COLOR', 'MAX_CATALOG_RI_COLOR',
                               'FIT_TRANSFORM', 'FIT_EXTINCTION', 'FIT_VIGNETTE', 'FIT_XY', 'FIT_JD']
FOCUS_LENGTH_MAX_PCT_DEVIATION = 5.0


# def remove_platesolution(mp_directory_path=None):
#     """ From FITS file, remove all header entries (except HISTORY lines) that support plate solution
#         previously written into the file. This is in preparation for re-solving the file with different
#         software: whole-image WCS plate solutions for now...one hopes for SIP-enabled plate solutions
#         in the future.
#         In any case, this unsolve-resolve work is made necessary by one bit of software that won't
#         play nicely with *any* other software.
#     :param mp_directory_path: path of the directory holding the MP FITS files to process.
#     :return: [None]. FITS files are updated in place.
#     """
#     mp_filenames = get_mp_filenames(mp_directory_path)
#     for fn in mp_filenames:
#         fullpath = os.path.join(mp_directory_path, fn)  # open to read only.
#         with apyfits.open(fullpath, mode='update') as hdulist:
#             hdu = hdulist[0]
#             header_keys = list(hdu.header.keys())
#             for key in PLATESOLUTION_KEYS_TO_REMOVE:
#                 if key.endswith('*'):
#                     for hk in header_keys:
#                         if hk.startswith(key[:-1]):
#                             try:
#                                 _ = int(hk[len(key[:-1]):])  # a test only.
#                             except ValueError:
#                                 pass  # if key doesn't end in an integer, do nothing.
#                             else:
#                                 del hdu.header[hk]  # if key does end in integer.
#                 if key in header_keys:
#                     del hdu.header[key]
#             hdu.header['HISTORY'] = 'Plate solution removed by remove_platesolution().'
#             hdulist.flush()  # finish writing to same file in same location.
#     print('OK: remove_platesolution() has processed', len(mp_filenames), 'MP FITS files.')


def start(mp_top_directory=MP_TOP_DIRECTORY, mp_id=None, an=None, filter=None):
    # Adapted from package mpc, mp_phot.start().
    """ Launch one session of MP photometry workflow.
    :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
               'C:/Astro/MP Photometry'. [string]
    :param mp_id: either a MP number, e.g., 1602 for Indiana [integer or string], or for an id string
               for unnumbered MPs only, e.g., ''. [string only]
    :param an: Astronight string representation, e.g., '20191106'. [integer or string]
    :param filter: name of filter for this session, or None to use default from instrument file. [string]
    :return: [None]
    """
    if mp_id is None or an is None:
        print(' >>>>> Usage: start(top_directory, mp_id, an)')
        return
    mp_id, an_string = process_mp_and_an(mp_id, an)

    # Construct directory path and make it the working directory:
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_id[1:], 'AN' + an_string)
    os.chdir(mp_directory)
    print('Working directory set to:', mp_directory)

    # Initiate log file and finish:
    log_file = open(LOG_FILENAME, mode='w')  # new file; wipe old one out if it exists.
    log_file.write(mp_directory + '\n')
    log_file.write('MP: ' + mp_id + '\n')
    log_file.write('AN: ' + an_string + '\n')
    log_file.write('FILTER:' + filter + '\n')
    log_file.write('This log started: ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    log_file.close()
    print('Log file started.')
    print('Next: assess()')


def resume(mp_top_directory=MP_TOP_DIRECTORY, mp_id=None, an=None, filter_string=None):
    # Adapted from package mpc, mp_phot.assess().
    """ Restart a workflow in its correct working directory,
        but keep the previous log file--DO NOT overwrite it.
    Parameters are exactly as for start().
    :return: [None]
    """
    if mp_id is None or an is None:
        print(' >>>>> Usage: resume(top_directory, mp_id, an)')
        return
    mp_id, an_string = process_mp_and_an(mp_id, an)

    # Construct directory path and make it the working directory:
    this_directory = os.path.join(mp_top_directory, 'MP_' + mp_id[1:], 'AN' + an_string)
    os.chdir(this_directory)

    # Verify that proper log file already exists in the working directory:
    this_context = get_context()
    if get_context() is None:
        print(' >>>>> Can\'t resume in', this_directory, '(has start() been run?)')
        return
    log_this_directory, log_mp_string, log_an_string, log_filter_string = this_context
    if log_mp_string.lower() == mp_id[1].lower() and \
            log_an_string.lower() == an_string.lower() and \
            log_filter_string == filter_string:
        print('READY TO GO in', this_directory)
    else:
        print(' >>>>> Can\'t resume in', this_directory)


def assess():
    # Adapt from package mpc, mp_phot.asses().
    """  First, verify that all required files are in the working directory or otherwise accessible.
         Then, perform checks on FITS files in this directory before performing the photometry proper.
         Modeled after and extended from assess() found in variable-star photometry package 'photrix'.
                                    May be zero for MP color index determination only. [int]
    :return: [None]
    """
    print('assess() entered.')
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string, filter_string = context
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    log_file.write('\n===== access()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Count FITS files by filter, write totals
    #    (we've stopped classifying files by intention; now we include all valid FITS in dfs):
    filter_counter = Counter()
    valid_fits_filenames = []
    all_fits_filenames = get_mp_filenames(this_directory)
    for filename in all_fits_filenames:
        fullpath = os.path.join(this_directory, filename)
        try:
            hdu = apyfits.open(fullpath)[0]
        except:
            print(' >>>>> WARNING: can\'t read file', fullpath, 'as FITS. Skipping file.')
        else:
            fits_filter = fits_header_value(hdu, 'FILTER')
            if fits_filter is None:
                print(' >>>>> WARNING: filter in', fullpath, 'cannot be read. Skipping file.')
            else:
                valid_fits_filenames.append(filename)
                filter_counter[fits_filter] += 1
    for filter in filter_counter.keys():
        print('   ' + str(filter_counter[filter]), 'in filter', filter + '.')

    # Start dataframe for main FITS integrity checks:
    fits_extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in valid_fits_filenames])
    df = pd.DataFrame({'Filename': valid_fits_filenames,
                       'Extension': fits_extensions.values}).sort_values(by=['Filename'])
    df = df.set_index('Filename', drop=False)
    df['PlateSolved'] = False
    df['Calibrated'] = True
    df['FWHM'] = np.nan
    df['FocalLength'] = np.nan

    # Populate df with FITS header info needed for validity tests below:
    for filename in df.index:
        fullpath = os.path.join(this_directory, filename)
        hdu = apyfits.open(fullpath)[0]  # already known to be valid, from above.
        df.loc[filename, 'PlateSolved'] = fits_is_plate_solved(hdu)
        # TODO: if is plate solved, add FITS header line 'plate solution verified' etc.
        # TODO: if is plate solved, calc and add any customary/PinPoint plate-sol lines back in.
        df.loc[filename, 'Calibrated'] = fits_is_calibrated(hdu)
        df.loc[filename, 'FWHM'] = fits_header_value(hdu, 'FWHM')
        df.loc[filename, 'FocalLength'] = fits_focal_length(hdu)
        jd_start = fits_header_value(hdu, 'JD')
        exposure = fits_header_value(hdu, 'EXPOSURE')
        jd_mid = jd_start + (exposure / 2) / 24 / 3600
        df.loc[filename, 'JD_mid'] = jd_mid  # needed only to write control.txt stub.
    n_warnings = 0

    # Warn of FITS without plate solution:
    filenames_not_platesolved = df.loc[~ df['PlateSolved'], 'Filename']
    if len(filenames_not_platesolved) >= 1:
        print('NO PLATE SOLUTION:')
        for fn in filenames_not_platesolved:
            print('    ' + fn)
        print('\n')
    else:
        print('All platesolved.')
    n_warnings += len(filenames_not_platesolved)

    # Warn of FITS without calibration:
    filenames_not_calibrated = df.loc[~ df['Calibrated'], 'Filename']
    if len(filenames_not_calibrated) >= 1:
        print('\nNOT CALIBRATED:')
        for fn in filenames_not_calibrated:
            print('    ' + fn)
        print('\n')
    else:
        print('All calibrated.')
    n_warnings += len(filenames_not_calibrated)

    # Warn of FITS with very large or very small FWHM:
    odd_fwhm_list = []
    settings = Settings()
    min_fwhm = 0.5 * settings['FWHM_NOMINAL']
    max_fwhm = 2.0 * settings['FWHM_NOMINAL']
    for fn in df['Filename']:
        fwhm = df.loc[fn, 'FWHM']
        if fwhm < min_fwhm or fwhm > max_fwhm:  # too small or large:
            odd_fwhm_list.append((fn, fwhm))
    if len(odd_fwhm_list) >= 1:
        print('\nUnusual FWHM (in pixels):')
        for fn, fwhm in odd_fwhm_list:
            print('    ' + fn + ' has unusual FWHM of ' + '{0:.2f}'.format(fwhm) + ' pixels.')
        print('\n')
    else:
        print('All FWHM values seem OK.')
    n_warnings += len(odd_fwhm_list)

    # Warn of FITS with abnormal Focal Length:
    odd_fl_list = []
    median_fl = df['FocalLength'].median()
    for f in df['Filename']:
        fl = df.loc[f, 'FocalLength']
        focus_length_pct_deviation = 100.0 * abs((fl - median_fl)) / median_fl
        if focus_length_pct_deviation > FOCUS_LENGTH_MAX_PCT_DEVIATION:
            odd_fl_list.append((f, fl))
    if len(odd_fl_list) >= 1:
        print('\nUnusual FocalLength (vs median of ' + '{0:.1f}'.format(median_fl) + ' mm:')
        for f, fl in odd_fl_list:
            print('    ' + f + ' has unusual Focal length of ' + str(fl))
        print('\n')
    else:
        print('All Focal Lengths seem OK.')
    n_warnings += len(odd_fl_list)

    # Summarize and write instructions for user's next steps:
    if n_warnings == 0:
        print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
        print('Next: (1) enter MP pixel positions in', CONTROL_FILENAME,
              'AND SAVE it,\n      (2) measure_mp()')
        log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
    else:
        print('\n >>>>> ' + str(n_warnings) + ' warnings (see listing above).')
        print('        Correct these and rerun assess() until no warnings remain.')
        log_file.write('assess(): ' + str(n_warnings) + ' warnings.' + '\n')
    log_file.close()

    write_control_txt_stub(this_directory, df)  # if it doesn't already exist.


def measure():
    # This will (1) prepare data for and call measure_mp() and measure_comps() from measure.py, then
    # (2) prepare the 3 key dataframes for the call to do_mp_phot().
    """ Prototype for testing; must be adapted deeply before production use. """
    MP_PHOT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_SESSIONS_DIRECTORY = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'test', '$sessions_for_test')
    TEST_MP = '191'
    TEST_AN = '20200617'
    mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    from test.test_do_workflow import make_test_control_txt
    make_test_control_txt()
    control_data = Control()
    ref_star_locations = [['MP_191-0001-Clear.fts',  790.6, 1115.0],
                          ['MP_191-0028-Clear.fts', 1198.5, 1084.4]]  # 28: close but faint
    mp_locations = control_data['MP_LOCATION']
    settings = Settings()

    # MP (minor planet) stack of calls:
    imlist = MP_ImageList.from_fits(mp_directory, TEST_MP, TEST_AN, 'Clear',
                                    ref_star_locations, mp_locations, settings)
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
    subarray_list.do_mp_aperture_photometry()
    df_mp_only = subarray_list.make_df_mp_only()
    # subarray_list.convolve_full_arrays([mpi.image.data for mpi in imlist.mp_images[0:1]])

    # Comp-star stack of calls:
    pass

    # Align and combine MP and comp-star dataframes, write them to csv files:
    pass


def reduce():
    # This will call do_mp_phot() in reduce.py.
    pass


CLASS_DEFINITIONS__________________________________________ = 0


class Settings:
    """ Holds (1) instrument info and (2) defaults for workflow in one dictionary 'data'. [dict]
        Supplied by user in text file stored at DEFAULTS_FILE_FULLPATH.
        Most items remain unchanged from night to night.
        Expected items:
            INSTRUMENT: name of instrument, e.g., 'Borea' [string]
            MP_RI_COLOR: default MP color in Sloan r-i, e.g. +0.22 [float]
            MIN_CATALOG_R_MAG: default minimum catalog comp-star mag in Sloan r [float]
            MAX_CATALOG_R_MAG: default maximum catalog comp-star mag in Sloan r [float]
            MAX_CATALOG_DR_MMAG: default maximum catalog comp-star mag uncertainty in Sloan r [float]
            MIN_CATALOG_RI_COLOR: default minimum catalog comp-star color in Sloan r-i [float]
            MAX_CATALOG_RI_COLOR: default maximum catalog comp-star color in Sloan r-i [float]
            FIT_TRANSFORM: True to include transform in comp-star regression model [boolean]
            FIT_EXTINCTION: True to include extinction in comp-star regression model [boolean]
            FIT_VIGNETTE: True to include parabolic vignetting in comp-star regression model [boolean]
            FIT_XY: True to include linear X- and Y-gradients in comp-star regression model [boolean]
            FIT_JD: True to include linear time term (drift) in comp-star regression model [boolean]
            PIXEL_SHIFT_TOLERANCE: maximum shift expected between images, in number of pixels [float]
            FWHM_NOMINAL: nominal comp-star full-width at half-max, to get computations started [float]
            CCD_GAIN: in electrons per ADU, e.g., 1.57 [float]
            ADU_SATURATION: ADU/pixel above which signals is expected saturated, e.g., 54000 [float]
            DEFAULT_FILTER: default MP filter, e.g., 'Clear' [string]
            PINPOINT_PIXEL_SCALE_FACTOR: real vs Pinpoint center pixel scale, e.g., 0.997 [float]
            TRANSFORM: list of default transform directives, [list of lists] e.g.,
                [['Clear', 'SR', 'SR-SI', 'Use', '+0.36', '-0.54'], ['BB', 'SR', 'SR-SI', 'Fit=2']]
    """
    def __init__(self, instrument_name=None):
        # Read and parse defaults.txt (only needed to construct control.txt stub):
        defaults_dict = dict_from_directives_file(DEFAULTS_FILE_FULLPATH)
        defaults_dict_directives = set(defaults_dict.keys())
        required_directives = set(REQUIRED_DEFAULT_DIRECTIVES)
        missing_directives = required_directives - defaults_dict_directives
        if len(missing_directives) > 0:
            for md in missing_directives:
                print(' >>>>> ERROR: defaults file is missing directive', md + '.')
            exit(0)
        if instrument_name is not None:
            defaults_dict['INSTRUMENT'] = instrument_name

        # Read and parse instrument file (given in __init__() or in defaults.txt):
        if instrument_name is None:
            self.instrument_filename = defaults_dict['INSTRUMENT'].split('.txt')[0] + '.txt'  # 1 .txt.
        else:
            self.instrument_filename = instrument_name.split('.txt')[0] + '.txt'  # 1 .txt @ end.
        fullpath = os.path.join(INSTRUMENT_FILE_DIRECTORY, self.instrument_filename)
        instrument_dict = dict_from_directives_file(fullpath)
        instrument_dict_directives = set(instrument_dict.keys())
        required_directives = set(REQUIRED_INSTRUMENT_DIRECTIVES)
        missing_directives = required_directives - instrument_dict_directives
        if len(missing_directives) > 0:
            for md in missing_directives:
                print(' >>>>> ERROR: instrument file is missing directive', md + '.')
            exit(0)

        # Verify no keys overlap, then combine dicts:
        directives_in_both_dicts = defaults_dict_directives.intersection(instrument_dict_directives)
        if len(directives_in_both_dicts) > 0:
            for db in directives_in_both_dicts:
                print(' >>>>> WARNING: directive', db, 'appears in both directive and instrument files.')
        self._data = {**defaults_dict, **instrument_dict}  # nb: instrument_dict overrides defaults_dict.

        # Cast to floats those values that actually represent floats:
        for key in ['MP_RI_COLOR', 'MIN_CATALOG_R_MAG', 'MAX_CATALOG_R_MAG',
                    'MAX_CATALOG_DR_MMAG', 'MIN_CATALOG_RI_COLOR', 'MAX_CATALOG_RI_COLOR',
                    'PIXEL_SHIFT_TOLERANCE', 'FWHM_NOMINAL', 'CCD_GAIN', 'ADU_SATURATION',
                    'PINPOINT_PIXEL_SCALE_FACTOR']:
            self._data[key] = float(self._data[key])

        # Split long transform strings into value lists:
        self._data['FIT_TRANSFORM'] = self._data['FIT_TRANSFORM'].split()
        if not isinstance(self._data['TRANSFORM'], list):
            self._data['TRANSFORM'] = [self._data['TRANSFORM']]
        self._data['TRANSFORM'] = [tr.split() for tr in self._data['TRANSFORM']]

    def __str__(self):
        return 'Settings object from instrument file ' + self.instrument_filename

    # Allow direct access as settings=Settings(); value = settings['somekey'].
    def __getitem__(self, key):
        return self._data.get(key, None)  # return None if key absent.

    def get(self, key, default_value):
        return self._data.get(key, default_value)


class Control:
    """ Holds data from control.txt file. Assumes current working directory is set by start().
        Supplied by user in text file 'control.txt' within session directory.
        Modified by user to control actual data reduction, including comp start selection
            and the fit model, including terms to include, MP color, and transform type.
        Expected items:
        FIT_EXTINCTION: True if user includes extinction in comp-star regression model. [boolean]
        FIT_JD: True if user includes linear time term (drift) in comp-star regression model. [boolean]
        FIT_TRANSFORM: True if user includes transform in comp-star regression model. [boolean]
        FIT_VIGNETTE: True if user includes parabolic vignetting in comp-star regression model. [boolean]
        FIT_XY: True if user includes linear X- and Y-gradients in comp-star regression model. [boolean]
        MAX_CATALOG_DR_MMAG: session's maximum catalog comp-star mag uncertainty in Sloan r [float]
        MAX_CATALOG_R_MAG: session's maximum catalog comp-star mag in Sloan r [float]
        MIN_CATALOG_R_MAG: session's maximum catalog comp-star mag in Sloan r [float]
        MAX_CATALOG_RI_COLOR: session's default maximum catalog comp-star color in Sloan r-i [float]
        MIN_CATALOG_RI_COLOR: session's default minimum catalog comp-star color in Sloan r-i [float]
        MP_RI_COLOR: user's session MP color in Sloan r-i, e.g. +0.22 [float]
        MP_LOCATION: 2-list of MP location specifications, from an early and a late image of the session,
            e.g., [['MP_191-0001-Clear.fts', 826.4, 1077.4], ['MP_191-0028-Clear.fts', 1144.3, 1099.3]]
        REF_STAR_LOCATION: list of at least 2 ref star location specifications, usually in the same image,
            e.g., [['MP_191-0001-Clear.fts', 790.6, 1115.0],
                   ['MP_191-0001-Clear.fts', 819.3, 1011.7],
                   ['MP_191-0001-Clear.fts', 1060.4, 1066.0]]
        IS_VALID: True if file 'control.txt' parsed without errors. [boolean]
        ERRORS: Errors encountered while parsing file 'control.txt'; typically []. [list of strings]
    """
    def __init__(self):
        context = get_context()
        if context is None:
            return
        this_directory, mp_string, an_string, filter_string = context
        self.fullpath = os.path.join(this_directory, CONTROL_FILENAME)
        control_dict = dict_from_directives_file(self.fullpath)
        control_directives = set(control_dict.keys())
        required_directives = set(REQUIRED_CONTROL_DIRECTIVES)
        missing_directives = required_directives - control_directives
        control_dict['IS_VALID'] = True
        control_dict['ERRORS'] = []
        if len(missing_directives) > 0:
            for md in missing_directives:
                print(' >>>>> ERROR: control file is missing directive', md + '.')
                control_dict['IS_VALID'] = False
                control_dict['ERRORS'].append('Missing directive: ' + md)

        # Verify at least 2 REF_STAR_LOCATION and exactly 2 MP_LOCATION entries:
        rsl = control_dict['REF_STAR_LOCATION']
        if not len(rsl) >= 2:
            print(' >>>>> ERROR: only', str(len(rsl)), 'REF_STAR_LOCATION lines, but >= 2 required.')
            control_dict['IS_VALID'] = False
            control_dict['ERRORS'].append('REF_STAR_LOCATION count: ' + str(len(rsl)) + ' but >=2 required.')
        mpl = control_dict['MP_LOCATION']
        if not len(mpl) == 2:
            print(' >>>>> ERROR:', str(len(mpl)), 'MP_LOCATION lines, but exactly 2 required.')
            control_dict['IS_VALID'] = False
            control_dict['ERRORS'].append('MP_LOCATION count: ' + str(len(mpl)) +
                                          ' but exactly 2 required.')

        # Cast values into proper types:
        for key in ['MP_RI_COLOR', 'MIN_CATALOG_R_MAG', 'MAX_CATALOG_R_MAG',
                    'MAX_CATALOG_DR_MMAG', 'MIN_CATALOG_RI_COLOR', 'MAX_CATALOG_RI_COLOR']:
            try:
                control_dict[key] = float(control_dict[key])
            except ValueError:
                print(' >>>>> ERROR: non-numeric value ' + str(control_dict[key]) +
                      'for directive ' + key + '.')
                control_dict['IS_VALID'] = False
                control_dict['ERRORS'].append('non-numeric value ' + str(control_dict[key]) +
                                              'for directive ' + key + '.')
        for key in ['FIT_EXTINCTION', 'FIT_VIGNETTE', 'FIT_XY', 'FIT_JD']:
            control_dict[key] = True if control_dict[key].upper()[0] == 'Y' else False
        new_values = []
        for raw_string in control_dict['REF_STAR_LOCATION']:
            tokens = raw_string.strip().rsplit(maxsplit=2)
            if len(tokens) != 3:
                print(' >>>>> ERROR: bad syntax in REF_STAR_LOCATION entry ' + raw_string)
                control_dict['IS_VALID'] = False
                control_dict['ERRORS'].append('bad syntax in REF_STAR_LOCATION entry ' + raw_string)
            try:
                new_value = [tokens[0], float(tokens[1]), float(tokens[2])]
            except ValueError:
                print(' >>>>> ERROR: non-numeric in REF_STAR_LOCATION entry ' + raw_string)
                control_dict['IS_VALID'] = False
                control_dict['ERRORS'].append('non-numeric in REF_STAR_LOCATION entry ' + raw_string)
                new_value = [tokens[0], None, None]
            new_values.append(new_value)
        control_dict['REF_STAR_LOCATION'] = new_values
        new_values = []
        for raw_string in control_dict['MP_LOCATION']:
            tokens = raw_string.strip().rsplit(maxsplit=2)
            if len(tokens) != 3:
                print(' >>>>> ERROR: bad syntax in MP_LOCATION entry ' + raw_string)
                control_dict['IS_VALID'] = False
                control_dict['ERRORS'].append('bad syntax in MP_LOCATION entry ' + raw_string)
            try:
                new_value = [tokens[0], float(tokens[1]), float(tokens[2])]
            except ValueError:
                print(' >>>>> ERROR: non-numeric in MP_LOCATION entry ' + raw_string)
                control_dict['IS_VALID'] = False
                control_dict['ERRORS'].append('non-numeric in MP_LOCATION entry ' + raw_string)
                new_value = [tokens[0], None, None]
            new_values.append(new_value)
        control_dict['MP_LOCATION'] = new_values
        self._data = control_dict

    def __str__(self):
        return 'Control object from ' + self.fullpath

    # Allow direct access as control=Control(); value = control['somekey'].
    def __getitem__(self, key):
        return self._data.get(key, None)  # return None if key absent.


SUPPORT_FUNCTIONS________________________________________________ = 0


def process_mp_and_an(mp_id, an):
    """ Return MP id and Astronight id in usable (internal) forms.
    :param mp_id: raw MP identification, either number or unnumbered ID. [int or string]
    :param an: Astronight ID yyyymmdd. [int or string]
    :return: 2-tuple (mp_id, an_string) [tuple of 2 strings]:
             mp_id has '#' prepended for numbered MP, '*' prepended for ID of unnumbered MP.
             an_string is always an 8 character string 'yyyymmdd' representing Astronight.
    """
    # Handle mp_id:
    if isinstance(mp_id, int):
        mp_id = ('#' + str(mp_id))  # mp_id like '#1108' for numbered MP 1108 (if passed in as int).
    else:
        if isinstance(mp_id, str):
            try:
                _ = int(mp_id)  # a test only
            except ValueError:
                mp_id = '*' + mp_id   # mp_id like '*1997 TX3' for unnumbered MP ID.
            else:
                mp_id = '#' + mp_id  # mp_id like '#1108' for numbered MP 1108 (if passed in as string).
        else:
            print(' >>>>> ERROR: mp_id must be an int or string')
            return None
    print('MP ID =', mp_id + '.')

    # Handle an:
    an_string = str(an).strip()
    try:
        _ = int(an_string)  # test only
    except ValueError:
        print(' >>>>> ERROR: an must be an int, or a string representing an integer.')
        return None
    print('AN =', an_string)
    return mp_id, an_string


def get_context():
    """ This is run at beginning of workflow functions (except start() or resume()) to orient the function.
    :return: 4-tuple: (this_directory, mp_string, an_string, filter_string) [4 strings]
    """
    this_directory = os.getcwd()
    if not os.path.isfile(LOG_FILENAME):
        print(' >>>>> ERROR: no log file found ==> You probably need to run start() or resume().')
        return None
    log_file = open(LOG_FILENAME, mode='r')  # for read only
    lines = log_file.readlines()
    log_file.close()
    if len(lines) < 3:
        return None
    if lines[0].strip().lower().replace('\\', '/').replace('//', '/') != \
            this_directory.strip().lower().replace('\\', '/').replace('//', '/'):
        print('Working directory does not match directory at top of log file.')
        return None
    mp_string = lines[1][3:].strip().upper()
    an_string = lines[2][3:].strip()
    filter_string = lines[3][7:].strip()
    return this_directory, mp_string, an_string, filter_string


def fits_header_value(hdu, key):
    # Adapted from package photrix, class FITS.header_value.
    """
    :param hdu: astropy fits header/data unit object.
    :param key: FITS header key [string] or list of keys to try [list of strings]
    :return: value of FITS header entry, typically [float] if possible, else [string]
    """
    if isinstance(key, str):  # case of single key to try.
        return hdu.header.get(key, None)
    for k in key:             # case of list of keys to try.
        value = hdu.header.get(k, None)
        if value is not None:
            return value
    return None


def fits_is_plate_solved(hdu):
    # Adapted loosely from package photrix, class FITS. Returns boolean.
    # TODO: tighten these tests, prob. by checking for reasonable numerical values.
    plate_solution_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']
    values = [fits_header_value(hdu, key) for key in plate_solution_keys]
    for v in values:
        if v is None:
            return False
        if not isinstance(v, float):
            try:
                _ = float(v)
            except ValueError:
                return False
            except TypeError:
                return False
    return True


def fits_is_calibrated(hdu):
    # Adapted from package photrix, class FITS. Requires calibration by Maxim 5 or 6, or by TheSkyX.
    # Returns boolean.
    # First, define all calibration functions as internal, nested functions:
    def _is_calibrated_by_maxim_5_or_6(hdu):
        calibration_value = fits_header_value(hdu, 'CALSTAT')
        if calibration_value is not None:
            if calibration_value.strip().upper() == 'BDF':
                return True
        return False
    # If any is function signals valid, then fits is calibrated:
    is_calibrated_list = [_is_calibrated_by_maxim_5_or_6(hdu)]  # expand later if more calibration fns.
    return any([is_cal for is_cal in is_calibrated_list])


def fits_focal_length(hdu):
    # Adapted from package photrix, class FITS. Returns float if valid, else None.
    fl = fits_header_value(hdu, 'FOCALLEN')
    if fl is not None:
        return fl  # in millimeters.
    x_pixel_size = fits_header_value(hdu, 'XPIXSZ')
    y_pixel_size = fits_header_value(hdu, 'YPIXSZ')
    x_pixel_scale = fits_header_value(hdu, 'CDELT1')
    y_pixel_scale = fits_header_value(hdu, 'CDELT2')
    if any([value is None for value in [x_pixel_size, y_pixel_size, x_pixel_scale, y_pixel_scale]]):
        return None
    fl_x = x_pixel_size / abs(x_pixel_scale) * (206265.0 / (3600 * 1000))
    fl_y = y_pixel_size / abs(y_pixel_scale) * (206265.0 / (3600 * 1000))
    return (fl_x + fl_y) / 2.0


def write_control_txt_stub(this_directory, df):
    # Prepare data required to write control.txt stub:
    defaults = Settings()
    jd_min = df['JD_mid'].min()
    df['SecondsRelative'] = [24 * 3600 * (jd - jd_min) for jd in df['JD_mid']]
    i_earliest = df['SecondsRelative'].nsmallest(n=1).index[0]
    i_latest = df['SecondsRelative'].nlargest(n=1).index[0]
    earliest_filename = df.loc[i_earliest, 'Filename']
    latest_filename = df.loc[i_latest, 'Filename']

    def yes_no(true_false):
        return 'Yes' if true_false else 'No'

    # Write file stub:
    lines = [';----- This is ' + CONTROL_FILENAME + ' for directory:\n;      ' + this_directory,
             ';',
             ';===== REF STAR LOCATIONS BLOCK ==========================================',
             ';===== Enter at least 2 in the SAME image, before measure_mp() ===========',
             ';      Reference Star x,y positions for image alignment:',
             '#REF_STAR_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; ',
             '#REF_STAR_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; ',
             '#REF_STAR_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; ',
             ';',
             ';===== MP LOCATIONS BLOCK ================================================',
             ';===== Enter exactly 2 in widely spaced images, before measure_mp() ======',
             ';      MP x,y positions for flux measurement:',
             '#MP_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
             'early filename, change if needed',
             '#MP_LOCATION  ' + latest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
             ' late filename, change if needed',
             ';',
             ';===== MP RI COLOR BLOCK =================================================',
             ';===== Enter before do_mp_phot(), get from do_color. =====================',
             '#MP_RI_COLOR ' + '{0:+.3f}'.format(defaults['MP_RI_COLOR']) +
             ' ;  get by running do_color(), or leave as default=' +
             '{0:+.3f}'.format(defaults['MP_RI_COLOR']),
             ';',
             ';===== SELECTION CRITERIA BLOCK ==========================================',
             ';===== Enter before do_mp_phot() =========================================',
             ';      Selection criteria for comp stars, observations, images:',
             ';#COMP  nnnn nn,   nnn        ; to omit comp(s) by comp ID',
             ';#OBS nnn,nnnn nnnn   nn      ; to omit observation(s) by Serial number',
             ';#IMAGE  MP_mmmm-00nn-Clear   ; to omit one FITS image (.fts at end optional)',
             ('#MIN_CATALOG_R_MAG ' + str(defaults['MIN_CATALOG_R_MAG'])).ljust(30) +
             '; default=' + str(defaults['MIN_CATALOG_R_MAG']),
             ('#MAX_CATALOG_R_MAG ' + str(defaults['MAX_CATALOG_R_MAG'])).ljust(30) +
             '; default=' + str(defaults['MAX_CATALOG_R_MAG']),
             ('#MAX_CATALOG_DR_MMAG ' + str(defaults['MAX_CATALOG_DR_MMAG'])).ljust(30) +
             '; default=' + str(defaults['MAX_CATALOG_DR_MMAG']),
             ('#MIN_CATALOG_RI_COLOR ' + str(defaults['MIN_CATALOG_RI_COLOR'])).ljust(30) +
             '; default=' + str(defaults['MIN_CATALOG_RI_COLOR']),
             ('#MAX_CATALOG_RI_COLOR ' + str(defaults['MAX_CATALOG_RI_COLOR'])).ljust(30) +
             '; default=' + str(defaults['MAX_CATALOG_RI_COLOR']),
             ';',
             ';===== REGRESSION OPTIONS BLOCK ==========================================',
             ';===== Enter before do_mp_phot(): ========================================',
             ';----- OPTIONS for regression model, rarely used:',

             ';Choices for #FIT_TRANSFORM: Fit=1; '
             + 'Fit=2; Use 0.2 0.4 [=tr1 & tr2 values]; Yes->Fit=1; No->Use 0 0',
             '#FIT_TRANSFORM  Fit=2'.ljust(30) + '; default= Fit=2',
             ('#FIT_EXTINCTION ' + yes_no(defaults['FIT_EXTINCTION'])).ljust(30) +
             '; default='
             + yes_no(defaults['FIT_EXTINCTION']) + ' // choose Yes or No  (case-insensitive)',
             ('#FIT_VIGNETTE ' + yes_no(defaults['FIT_VIGNETTE'])).ljust(30) + '; default='
             + yes_no(defaults['FIT_VIGNETTE']) + ' // choose Yes or No  (case-insensitive)',
             ('#FIT_XY ' + yes_no(defaults['FIT_XY'])).ljust(30) + '; default='
             + yes_no(defaults['FIT_XY']) + ' // choose Yes or No  (case-insensitive)',
             ('#FIT_JD ' + yes_no(defaults['FIT_JD'])).ljust(30) + '; default='
             + yes_no(defaults['FIT_JD']) + ' // choose Yes or No  (case-insensitive)',
             ';'
             ]
    lines = [line + '\n' for line in lines]
    fullpath = os.path.join(this_directory, CONTROL_FILENAME)
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
    print('New ' + CONTROL_FILENAME + ' file written.\n')


# def apply_pixel_scale_factor(fullpath):
#     hdu0 = apyfits.open(fullpath)[0]
#     h = hdu0.header
#     print(str(h['CDELT1']), str(h['CDELT2']))
#     cdelt1 = np.sign(h['CDELT1']) * (
#             (abs(h['CDELT1']) + abs(h['CDELT2'])) / 2.0) * PINPOINT_PLATE_FACTOR
#     cdelt2 = np.sign(h['CDELT2']) * (
#             (abs(h['CDELT1']) + abs(h['CDELT2'])) / 2.0) * PINPOINT_PLATE_FACTOR
#     cd11 = + cdelt1 * cos(h['CROTA2'] * RADIANS_PER_DEGREE)
#     cd12 = - cdelt2 * sin(h['CROTA2'] * RADIANS_PER_DEGREE)
#     cd21 = + cdelt1 * sin(h['CROTA2'] * RADIANS_PER_DEGREE)
#     cd22 = + cdelt2 * cos(h['CROTA2'] * RADIANS_PER_DEGREE)
#     h['CDELT1'] = cdelt1
#     h['CDELT2'] = cdelt2
#     h['CD1_1'] = cd11
#     h['CD1_2'] = cd12
#     h['CD2_1'] = cd21
#     h['CD2_2'] = cd22
#     print(str(h['CDELT1']), str(h['CDELT2']))
#     radesysa = h.get('RADECSYS')  # replace obsolete key if present.
#     if radesysa is not None:
#         h['RADESYSa'] = radesysa
#         del h['RADECSYS']
#     h['HISTORY'] = PIXEL_FACTOR_HISTORY_TEXT
#     this_wcs = WCS(header=h)
#     ccddata_object = CCDData(data=hdu0.data, wcs=this_wcs, header=h, unit='adu')
#     return ccddata_object
