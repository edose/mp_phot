__author__ = "Eric Dose, Albuquerque"

# Python core packages:
import os
from datetime import datetime, timezone
from collections import Counter
from math import sqrt, cos, sin, pi, floor
import configparser

# External packages:
import numpy as np
import pandas as pd
import astropy.io.fits as apyfits
from astropy.nddata import CCDData
from astropy.wcs import WCS

# EVD packages:
from astropak.ini import IniFile
from astropak.legacy import FITS, Image
from astropak.util import jd_from_datetime_utc

# From this package:
from .bulldozer import MP_ImageList
import mp_phot.ini as ini
import mp_phot.util as util

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_TEMPLATE_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

RADIANS_PER_DEGREE = pi / 180.0
PIXEL_FACTOR_HISTORY_TEXT = 'Pixel scale factor applied by apply_pixel_scale_factor().'
FOCUS_LENGTH_MAX_PCT_DEVIATION = 5.0


def start(session_top_directory=None, mp_id=None, an=None, filter=None):
    # Adapted from package mpc, mp_phot.start().
    """ Launch one session of MP photometry workflow.
    :param session_top_directory: path of lowest directory common to all MP lightcurve FITS, e.g.,
               'C:/Astro/MP Photometry'. None will use .ini file default (normal case). [string]
    :param mp_id: either a MP number, e.g., 1602 for Indiana [integer or string], or for an id string
               for unnumbered MPs only, e.g., ''. [string only]
    :param an: Astronight string representation, e.g., '20191106'. [integer or string]
    :param filter: name of filter for this session, or None to use default from instrument file. [string]
    :return: [None]
    """
    if mp_id is None or an is None:
        print(' >>>>> Usage: start(top_directory, mp_id, an)')
        return
    mp_id, an_string = util.get_mp_and_an_strings(mp_id, an)

    defaults_dict = ini.make_defaults_dict()
    if session_top_directory is None:
        session_top_directory = defaults_dict['session top directory']

    # Construct directory path and make it the working directory:
    mp_directory = os.path.join(session_top_directory, 'MP_' + mp_id, 'AN' + an_string)
    os.chdir(mp_directory)
    print('Working directory set to:', mp_directory)

    # Initiate log file and finish:
    log_filename = defaults_dict['session log filename']
    with open(log_filename, mode='w') as log_file:  # new file; delete any old one.
        log_file.write('Session Log File.' + '\n')
        log_file.write(mp_directory + '\n')
        log_file.write('MP: ' + mp_id + '\n')
        log_file.write('AN: ' + an_string + '\n')
        log_file.write('FILTER:' + filter + '\n')
        log_file.write('This log started: ' +
                       '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    print('Log file started.')
    print('Next: assess()')


def resume(session_top_directory=None, mp_id=None, an=None, filter=None):
    # Adapted from package mpc, mp_phot.assess().
    """ Restart a workflow in its correct working directory,
        but keep the previous log file--DO NOT overwrite it.
    Parameters are exactly as for .start().
    :return: [None]
    """
    if mp_id is None or an is None:
        print(' >>>>> Usage: resume(top_directory, mp_id, an)')
        return
    mp_id, an_string = util.get_mp_and_an_strings(mp_id, an)

    defaults_dict = ini.make_defaults_dict()
    if session_top_directory is None:
        session_top_directory = defaults_dict['session top directory']

    # Construct directory path and make it the working directory:
    this_directory = os.path.join(session_top_directory, 'MP_' + mp_id, 'AN' + an_string)
    os.chdir(this_directory)

    # Verify that proper log file already exists in the working directory:
    this_context = get_context()
    if get_context() is None:
        print(' >>>>> Can\'t resume in', this_directory, '(has start() been run?)')
        return
    log_this_directory, log_mp_string, log_an_string, log_filter_string = this_context
    if log_mp_string.lower() == mp_id[1].lower() and \
            log_an_string.lower() == an_string.lower() and \
            log_filter_string == filter:
        print('READY TO GO in', this_directory)
    else:
        print(' >>>>> Can\'t resume in', this_directory)


def assess(return_results=False):
    """  First, verify that all required files are in the working directory or otherwise accessible.
         Then, perform checks on FITS files in this directory before performing the photometry proper.
         Modeled after and extended from assess() found in variable-star photometry package 'photrix',
             then adapted from package mpc, mp_phot.assess()
    :return: [None], or dict of summary info and warnings. [py dict]
    """
    # Setup, including initializing return_dict:
    # (Can't use orient_this_function(), because control.ini may not exist yet.)
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string, filter_string = context
    defaults_dict = ini.make_defaults_dict()

    return_dict = {
        'file not read': [],         # list of filenames
        'filter not read': [],       # "
        'file count by filter': [],  # list of tuples (filter, file count)
        'warning count': 0,          # total count of all warnings.
        'not platesolved': [],       # list of filenames
        'not calibrated': [],        # "
        'unusual fwhm': [],          # list of tuples (filename, fwhm)
        'unusual focal length': []}  # list of tuples (filename, focal length)

    # Count FITS files by filter, write totals
    #    (we've stopped classifying files by intention; now we include all valid FITS in dfs):
    filter_counter = Counter()
    valid_fits_filenames = []
    all_fits_filenames = util.get_mp_filenames(this_directory)
    for filename in all_fits_filenames:
        fullpath = os.path.join(this_directory, filename)
        try:
            hdu = apyfits.open(fullpath)[0]
        except:
            print(' >>>>> WARNING: can\'t read file', fullpath, 'as FITS. Skipping file.')
            return_dict['file not read'].append(filename)
        else:
            fits_filter = util.fits_header_value(hdu, 'FILTER')
            if fits_filter is None:
                print(' >>>>> WARNING: filter in', fullpath, 'cannot be read. Skipping file.')
                return_dict['filter not read'].append(filename)
            else:
                valid_fits_filenames.append(filename)
                filter_counter[fits_filter] += 1
    for filter in filter_counter.keys():
        print('   ' + str(filter_counter[filter]), 'in filter', filter + '.')
        return_dict['file count by filter'].append((filter, filter_counter[filter]))

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
        df.loc[filename, 'PlateSolved'] = util.fits_is_plate_solved(hdu)
        # TODO: if is plate solved, add FITS header line 'plate solution verified' etc.
        # TODO: if is plate solved, calc and add any customary/PinPoint plate-sol lines back in.
        df.loc[filename, 'Calibrated'] = util.fits_is_calibrated(hdu)
        df.loc[filename, 'FWHM'] = util.fits_header_value(hdu, 'FWHM')
        df.loc[filename, 'FocalLength'] = util.fits_focal_length(hdu)
        jd_start = util.fits_header_value(hdu, 'JD')
        exposure = util.fits_header_value(hdu, 'EXPOSURE')
        jd_mid = jd_start + (exposure / 2) / 24 / 3600
        df.loc[filename, 'JD_mid'] = jd_mid  # needed only to write control.ini stub.

    # Warn of FITS without plate solution:
    filenames_not_platesolved = df.loc[~ df['PlateSolved'], 'Filename']
    if len(filenames_not_platesolved) >= 1:
        print('NO PLATE SOLUTION:')
        for fn in filenames_not_platesolved:
            print('    ' + fn)
            return_dict['not platesolved'].append(fn)
        print('\n')
    else:
        print('All platesolved.')
    return_dict['warning count'] += len(filenames_not_platesolved)

    # Warn of FITS without calibration:
    filenames_not_calibrated = df.loc[~ df['Calibrated'], 'Filename']
    if len(filenames_not_calibrated) >= 1:
        print('\nNOT CALIBRATED:')
        for fn in filenames_not_calibrated:
            print('    ' + fn)
            return_dict['not calibrated'].append(fn)
        print('\n')
    else:
        print('All calibrated.')
    return_dict['warning count'] += len(filenames_not_calibrated)

    # Warn of FITS with very large or very small FWHM:
    odd_fwhm_list = []
    instrument_dict = ini.make_instrument_dict(defaults_dict)
    # settings = Settings()
    min_fwhm = 0.5 * instrument_dict['nominal fwhm pixels']
    max_fwhm = 2.0 * instrument_dict['nominal fwhm pixels']
    for fn in df['Filename']:
        fwhm = df.loc[fn, 'FWHM']
        if fwhm < min_fwhm or fwhm > max_fwhm:  # too small or large:
            odd_fwhm_list.append((fn, fwhm))
    if len(odd_fwhm_list) >= 1:
        print('\nUnusual FWHM (in pixels):')
        for fn, fwhm in odd_fwhm_list:
            print('    ' + fn + ' has unusual FWHM of ' + '{0:.2f}'.format(fwhm) + ' pixels.')
            return_dict['unusual fwhm'].append((fn, fwhm))
        print('\n')
    else:
        print('All FWHM values seem OK.')
    return_dict['warning count'] += len(odd_fwhm_list)

    # Warn of FITS with abnormal Focal Length:
    odd_fl_list = []
    median_fl = df['FocalLength'].median()
    for fn in df['Filename']:
        fl = df.loc[fn, 'FocalLength']
        focus_length_pct_deviation = 100.0 * abs((fl - median_fl)) / median_fl
        if focus_length_pct_deviation > FOCUS_LENGTH_MAX_PCT_DEVIATION:
            odd_fl_list.append((fn, fl))
    if len(odd_fl_list) >= 1:
        print('\nUnusual FocalLength (vs median of ' + '{0:.1f}'.format(median_fl) + ' mm:')
        for fn, fl in odd_fl_list:
            print('    ' + fn + ' has unusual Focal length of ' + str(fl))
            return_dict['unusual focal length'].append((fn, fl))
        print('\n')
    else:
        print('All Focal Lengths seem OK.')
    return_dict['warning count'] += len(odd_fl_list)

    # Summarize and write instructions for user's next steps:
    control_filename = defaults_dict['session control filename']
    session_log_filename = defaults_dict['session log filename']
    session_log_fullpath = os.path.join(this_directory, session_log_filename)
    with open(session_log_fullpath, mode='w') as log_file:
        if return_dict['warning count'] == 0:
            print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
            print('Next: (1) enter MP pixel positions in', control_filename,
                  'AND SAVE it,\n      (2) measure_mp()')
            log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
        else:
            print('\n >>>>> ' + str(return_dict['warning count']) + ' warnings (see listing above).')
            print('        Correct these and rerun assess() until no warnings remain.')
            log_file.write('assess(): ' + str(return_dict['warning count']) + ' warnings.' + '\n')

    df_temporal = df.loc[:, ['Filename', 'JD_mid']].sort_values(by=['JD_mid'])
    filenames_temporal_order = df_temporal['Filename']
    write_control_ini_stub(this_directory, filenames_temporal_order)  # if it doesn't already exist.
    if return_results:
        return return_dict


def make_df_images():
    """ This dataframe probably needed for following fns, may as well get its construction out of the way.
        Adapted 2021-01-10 from mpc.mp_phot.make_dfs().
    :return df_images: one row per valid FITS image in current working directory. [pandas dataframe]
    """
    context, defaults_dict, control_dict, log_file = orient_this_function('make_df_images')
    this_directory, mp_string, an_string, filter_string = context

    # Get all relevant FITS filenames, make lists of FITS objects and Image objects:
    all_fits_filenames = util.get_mp_filenames(this_directory)
    fits_list = [FITS(this_directory, '', fits_name) for fits_name in all_fits_filenames]
    valid_fits_list = [fits for fits in fits_list if fits.is_valid]
    image_list = [Image(fits_object) for fits_object in valid_fits_list]  # Image objects
    if len(image_list) <= 0:
        print(' >>>>> ERROR: no FITS files found in', this_directory + ' --> exiting now.')
        return None

    # Gather image data into a list of dicts (to then make dataframe):
    image_dict_list = []
    for image in image_list:
        image_dict = dict()
        image_dict['FITSfile'] = image.fits.filename
        image_dict['JD_start'] = jd_from_datetime_utc(image.fits.utc_start)
        image_dict['UTC_start'] = image.fits.utc_start
        image_dict['Exposure'] = image.fits.exposure
        image_dict['UTC_mid'] = image.fits.utc_mid
        image_dict['JD_mid'] = jd_from_datetime_utc(image.fits.utc_mid)
        image_dict['Filter'] = image.fits.filter
        image_dict['Airmass'] = image.fits.airmass
        image_dict['JD_fract'] = np.nan  # placeholder (actual value requires that all JDs be known).
        image_dict_list.append(image_dict)

    # Make df_images (one row per FITS file):
    df_images = pd.DataFrame(data=image_dict_list)
    df_images.index = list(df_images['FITSfile'])  # list to prevent naming the index
    jd_floor = floor(df_images['JD_mid'].min())  # requires that all JD_mid values be known.
    df_images['JD_fract'] = df_images['JD_mid'] - jd_floor
    df_images.sort_values(by='JD_mid')
    df_images = util.reorder_df_columns(df_images, ['FITSfile', 'JD_mid', 'Filter', 'Exposure', 'Airmass'])
    return df_images


def calc_mp_fluxes():
    """ By calling various fns from bulldozer.py: prepare MP fluxes (net) and make a dataframe with
        these fluxes, their sigmas, and probably some related data (which?).
        IN DEVELOPMENT 2021-01-04.
    :return df_mp_fluxes
    """
    context, defaults_dict, control_dict, log_file = orient_this_function('calc_mp_fluxes')
    this_directory, mp_string, an_string, filter_string = context

    # Defunct, previous-version:
    # MP_PHOT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # TEST_SESSIONS_DIRECTORY = os.path.join(MP_PHOT_ROOT_DIRECTORY, 'test', '$sessions_for_test')
    # TEST_MP = '191'
    # TEST_AN = '20200617'
    # mp_directory = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_' + TEST_MP, 'AN' + TEST_AN)
    # from test.XXX_test_do_workflow import make_test_control_txt
    # make_test_control_txt()
    # # control_data = Control()
    # given_ref_star_xy = [['MP_191-0001-Clear.fts',  790.6, 1115.0],
    #                       ['MP_191-0028-Clear.fts', 1198.5, 1084.4]]  # 28: close but faint
    # mp_locations = control_data['MP_LOCATION']
    # # settings = Settings()

    # TODO: Purge settings object, replace with control_dict.
    # MP (minor planet) stack of calls:
    imlist = MP_ImageList.from_fits(this_directory, mp_string, an_string, 'Clear', control_dict)
    imlist.calc_ref_star_radecs()
    imlist.calc_mp_radecs_and_xy()
    imlist.make_subimages()
    imlist.wcs_align_subimages()
    imlist.trim_nans_from_subimages()
    imlist.get_subimage_locations()
    subarray_list = imlist.make_subarrays()  # make simple numpy arrays, give up CCDData objects.

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


def calc_comp_star_fluxes():
    """

    :return:
    """
    pass

def calc_magnitudes():
    pass


def make_lightcurves():
    # This will call do_mp_phot() in reduce.py.
    pass


SUPPORT_FUNCTIONS________________________________________________ = 0


def get_context():
    """ This is run at beginning of workflow functions (except start() or resume()) to orient the function.
        TESTS OK 2021-01-08.
    :return: 4-tuple: (this_directory, mp_string, an_string, filter_string) [4 strings]
    """
    this_directory = os.getcwd()
    defaults_dict = ini.make_defaults_dict()
    session_log_filename = defaults_dict['session log filename']
    session_log_fullpath = os.path.join(this_directory, session_log_filename)
    log_file = open(session_log_fullpath, mode='r')
    if not os.path.isfile(session_log_fullpath):
        print(' >>>>> ERROR: no log file found ==> You probably need to run start() or resume().')
        return None
    with open(session_log_filename, mode='r') as log_file:
        lines = log_file.readlines()
    if len(lines) < 5:
        return None
    if not lines[0].lower().startswith('session log file'):
        return None
    if lines[1].strip().lower().replace('\\', '/').replace('//', '/') != \
            this_directory.strip().lower().replace('\\', '/').replace('//', '/'):
        print('Working directory does not match directory at top of log file.')
        return None
    mp_string = lines[2][3:].strip().upper()
    an_string = lines[3][3:].strip()
    filter_string = lines[4][7:].strip()
    return this_directory, mp_string, an_string, filter_string


def orient_this_function(calling_function_name='[FUNCTION NAME NOT GIVEN]'):
    """ Typically called at the top of workflow functions, to collect commonly required data.
    :return: tuple of data elements: context [tuple], defaults_dict [py dict], log_file [file object].
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string, filter_string = context
    defaults_dict = ini.make_defaults_dict()
    control_dict = make_control_dict()
    log_filename = defaults_dict['session log filename']
    log_file = open(log_filename, mode='a')  # set up append to log file.
    log_file.write('\n===== ' + calling_function_name + '()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    return context, defaults_dict, control_dict, log_file


def write_control_ini_stub(this_directory, filenames_temporal_order):
    """ Write the initial control file for this lightcurve session, later to be edited by user.
    :param this_directory:
    :param filenames_temporal_order:
    :return:
    """
    # Do not overwrite existing control file:
    defaults_dict = ini.make_defaults_dict()
    control_ini_filename = defaults_dict['session control filename']
    fullpath = os.path.join(this_directory, control_ini_filename)
    if os.path.exists(fullpath):
        return

    filename_earliest = filenames_temporal_order[0]
    filename_latest = filenames_temporal_order[-1]
    header_lines = [
        '# This is ' + fullpath + '.',
        '']

    ini_lines = [
        '[Ini Template]',
        'Filename = control.template',
        '']
    bulldozer_lines = [
        '[Bulldozer]',
        '# At least 3 ref star XY, one per line, all from one FITS only if at all possible:',
        'Ref Star XY = ' + filename_earliest + ' 000.0  000.0',
        '              ' + filename_earliest + ' 000.0  000.0',
        '              ' + filename_earliest + ' 000.0  000.0',
        '# Exactly 2 MP XY, one per line (typically earliest and latest FITS):',
        'MP XY = ' + filename_earliest + ' 000.0  000.0',
        '        ' + filename_latest + ' 000.0  000.0',
        '']
    selection_criteria_lines = [
        '[Selection Criteria]',
        'Omit Comps =',
        'Omit Obs =',
        'Omit Images =',
        'Min Catalog r mag = 10.0',
        'Max Catalog r mag = 16.0',
        'Max Catalog dr mmag = 15.0',
        'Min Catalog ri color = 0.04',
        'Max Catalog ri color = 0.40',
        '']
    regression_lines = [
        '[Regression]',
        'MP ri color = +0.220',
        '# Fit Transform, one of: Fit=1, Fit=2, Use [val1], Use [val1] [val2]:',
        'Fit Transform = Use +0.4 -0.6',
        '# Fit Extinction, one of: Yes, Use [val]:',
        'Fit Extinction = Use +0.16',
        'Fit Vignette = Yes',
        'Fit XY = No',
        'Fit JD = Yes']
    raw_lines = header_lines + ini_lines + bulldozer_lines + selection_criteria_lines + regression_lines
    ready_lines = [line + '\n' for line in raw_lines]
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(ready_lines)
    print('New ' + control_ini_filename + ' file written.\n')


def make_control_dict():
    """ Read the control file for this lightcurve session, return control dict.
    :return: control_dict [py dict].
    """
    this_context = get_context()
    this_directory, mp_string, an_string, filter_string = this_context
    defaults_dict = ini.make_defaults_dict()
    control_ini_filename = defaults_dict['session control filename']
    control_ini_directory = defaults_dict['session top directory']
    fullpath = os.path.join(this_directory, control_ini_filename)
    control_ini = IniFile(fullpath, template_directory_path=CONTROL_TEMPLATE_DIRECTORY)
    control_dict = control_ini.value_dict  # raw values, a few to be reparsed just below:

    # Bulldozer section:
    # Parse and overwrite 'ref star xy':
    ref_star_xy_list = []
    ref_star_xy_lines = [line.strip() for line in control_dict['ref star xy']]
    for line in ref_star_xy_lines:
        items = line.replace(',', ' ').rsplit(maxsplit=2)  # for each line, items are: filename x y
        if len(items) == 3:
            filename = items[0]
            x = ini.float_or_warn(items[1], filename + 'Ref Star X' + items[1])
            y = ini.float_or_warn(items[2], filename + 'Ref Star Y' + items[2])
            ref_star_xy_list.append((filename, x, y))
        elif len(items >= 1):
            print(' >>>>> ERROR: ' + items[1] + ' Ref Star XY invalid: ' + line)
            return None
    if len(ref_star_xy_list) < 2:
        print(' >>>>> ERROR: control \'ref star xy\' has fewer than 2 entries, not allowed.')
    control_dict['ref star xy'] = ref_star_xy_list

    # Parse and overwrite 'mp xy':
    mp_xy_list = []
    mp_xy_lines = [line.strip() for line in control_dict['mp xy']]
    for line in mp_xy_lines:
        items = line.replace(',', ' ').rsplit(maxsplit=2)  # for each line, items are: filename x y
        if len(items) == 3:
            filename = items[0]
            x = ini.float_or_warn(items[1], filename + 'MP X' + items[1])
            y = ini.float_or_warn(items[2], filename + 'MP X' + items[2])
            mp_xy_list.append((filename, x, y))
        elif len(items != 2):
            print(' >>>>> ERROR: ' + items[1] + ' MP XY invalid: ' + line)
            return None
    if len(mp_xy_list) != 2:
        print(' >>>>> ERROR: control \'ref star xy\' has ', str(len(mp_xy_list)),
              ' entries, but must have exactly 2.')
    control_dict['mp xy'] = mp_xy_list

    # Selection Criteria section, Omit elements:
    control_dict['omit comps'] = ini.multiline_ini_value_to_items(' '.join(control_dict['omit comps']))
    control_dict['omit obs'] = ini.multiline_ini_value_to_items(' '.join(control_dict['omit obs']))
    control_dict['omit images'] = [s.strip() for s in control_dict['omit images']]
    for comp in control_dict['omit comps']:
        ini.warn_if_not_positive_int(comp, 'Comp: ' + comp)
    for obs in control_dict['omit obs']:
        ini.warn_if_not_positive_int(obs, 'Obs: ' + obs)

    # Standardize remaining elements:
    ini.warn_if_not_float(control_dict['mp ri color'], 'control.ini: MP ri Color is not a float.')
    control_dict['fit transform'] = tuple([item.lower()
                                           for item in control_dict['fit transform'].split()])
    control_dict['fit extinction'] = tuple([item.lower()
                                            for item in control_dict['fit extinction'].split()])
    if len(control_dict['fit extinction']) == 1:
        control_dict['fit extinction'] = control_dict['fit extinction'][0]
    return control_dict
