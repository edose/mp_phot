__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

_____FROM_old_workflow_session_py_________________________ = 0

# ===== This was from workflow_session.py =====
# PLATESOLUTION_KEYS_TO_REMOVE = ['CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
#                                 'ZMAG', 'EPOCH', 'PA', 'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2',
#                                 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'TR1_*', 'TR2_*', 'PLTSOLVD']
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
# def write_control_txt_stub(this_directory, df):
#     # Prepare data required to write control.txt stub:
#     defaults = Settings()
#     jd_min = df['JD_mid'].min()
#     df['SecondsRelative'] = [24 * 3600 * (jd - jd_min) for jd in df['JD_mid']]
#     i_earliest = df['SecondsRelative'].nsmallest(n=1).index[0]
#     i_latest = df['SecondsRelative'].nlargest(n=1).index[0]
#     earliest_filename = df.loc[i_earliest, 'Filename']
#     latest_filename = df.loc[i_latest, 'Filename']
#
#     def yes_no(true_false):
#         return 'Yes' if true_false else 'No'
#
#     # Write file stub:
#     lines = [';----- This is ' + CONTROL_FILENAME + ' for directory:\n;      ' + this_directory,
#              ';',
#              ';===== REF STAR LOCATIONS BLOCK ==========================================',
#              ';===== Enter at least 2 in the SAME image, before measure_mp() ===========',
#              ';      Reference Star x,y positions for image alignment:',
#              '#REF_STAR_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; ',
#              '#REF_STAR_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; ',
#              '#REF_STAR_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; ',
#              ';',
#              ';===== MP LOCATIONS BLOCK ================================================',
#              ';===== Enter exactly 2 in widely spaced images, before measure_mp() ======',
#              ';      MP x,y positions for flux measurement:',
#              '#MP_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
#              'early filename, change if needed',
#              '#MP_LOCATION  ' + latest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
#              ' late filename, change if needed',
#              ';',
#              ';===== MP RI COLOR BLOCK =================================================',
#              ';===== Enter before do_mp_phot(), get from do_color. =====================',
#              '#MP_RI_COLOR ' + '{0:+.3f}'.format(defaults['MP_RI_COLOR']) +
#              ' ;  get by running do_color(), or leave as default=' +
#              '{0:+.3f}'.format(defaults['MP_RI_COLOR']),
#              ';',
#              ';===== SELECTION CRITERIA BLOCK ==========================================',
#              ';===== Enter before do_mp_phot() =========================================',
#              ';      Selection criteria for comp stars, observations, images:',
#              ';#COMP  nnnn nn,   nnn        ; to omit comp(s) by comp ID',
#              ';#OBS nnn,nnnn nnnn   nn      ; to omit observation(s) by Serial number',
#              ';#IMAGE  MP_mmmm-00nn-Clear   ; to omit one FITS image (.fts at end optional)',
#              ('#MIN_CATALOG_R_MAG ' + str(defaults['MIN_CATALOG_R_MAG'])).ljust(30) +
#              '; default=' + str(defaults['MIN_CATALOG_R_MAG']),
#              ('#MAX_CATALOG_R_MAG ' + str(defaults['MAX_CATALOG_R_MAG'])).ljust(30) +
#              '; default=' + str(defaults['MAX_CATALOG_R_MAG']),
#              ('#MAX_CATALOG_DR_MMAG ' + str(defaults['MAX_CATALOG_DR_MMAG'])).ljust(30) +
#              '; default=' + str(defaults['MAX_CATALOG_DR_MMAG']),
#              ('#MIN_CATALOG_RI_COLOR ' + str(defaults['MIN_CATALOG_RI_COLOR'])).ljust(30) +
#              '; default=' + str(defaults['MIN_CATALOG_RI_COLOR']),
#              ('#MAX_CATALOG_RI_COLOR ' + str(defaults['MAX_CATALOG_RI_COLOR'])).ljust(30) +
#              '; default=' + str(defaults['MAX_CATALOG_RI_COLOR']),
#              ';',
#              ';===== REGRESSION OPTIONS BLOCK ==========================================',
#              ';===== Enter before do_mp_phot(): ========================================',
#              ';----- OPTIONS for regression model, rarely used:',
#
#              ';Choices for #FIT_TRANSFORM: Fit=1; '
#              + 'Fit=2; Use 0.2 0.4 [=tr1 & tr2 values]; Yes->Fit=1; No->Use 0 0',
#              '#FIT_TRANSFORM  Fit=2'.ljust(30) + '; default= Fit=2',
#              ('#FIT_EXTINCTION ' + yes_no(defaults['FIT_EXTINCTION'])).ljust(30) +
#              '; default='
#              + yes_no(defaults['FIT_EXTINCTION']) + ' // choose Yes or No  (case-insensitive)',
#              ('#FIT_VIGNETTE ' + yes_no(defaults['FIT_VIGNETTE'])).ljust(30) + '; default='
#              + yes_no(defaults['FIT_VIGNETTE']) + ' // choose Yes or No  (case-insensitive)',
#              ('#FIT_XY ' + yes_no(defaults['FIT_XY'])).ljust(30) + '; default='
#              + yes_no(defaults['FIT_XY']) + ' // choose Yes or No  (case-insensitive)',
#              ('#FIT_JD ' + yes_no(defaults['FIT_JD'])).ljust(30) + '; default='
#              + yes_no(defaults['FIT_JD']) + ' // choose Yes or No  (case-insensitive)',
#              ';'
#              ]
#     lines = [line + '\n' for line in lines]
#     fullpath = os.path.join(this_directory, CONTROL_FILENAME)
#     if not os.path.exists(fullpath):
#         with open(fullpath, 'w') as f:
#             f.writelines(lines)
#     print('New ' + CONTROL_FILENAME + ' file written.\n')


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
#     def __init__(self, instrument_name=None):
#         # Read and parse defaults.txt (only needed to construct control.txt stub):
#         defaults_dict = dict_from_directives_file(DEFAULTS_FILE_FULLPATH)
#         defaults_dict_directives = set(defaults_dict.keys())
#         required_directives = set(REQUIRED_DEFAULT_DIRECTIVES)
#         missing_directives = required_directives - defaults_dict_directives
#         if len(missing_directives) > 0:
#             for md in missing_directives:
#                 print(' >>>>> ERROR: defaults file is missing directive', md + '.')
#             exit(0)
#         if instrument_name is not None:
#             defaults_dict['INSTRUMENT'] = instrument_name
#
#         # Read and parse instrument file (given in __init__() or in defaults.txt):
#         if instrument_name is None:
#             self.instrument_filename = defaults_dict['INSTRUMENT'].split('.txt')[0] + '.txt'  # 1 .txt.
#         else:
#             self.instrument_filename = instrument_name.split('.txt')[0] + '.txt'  # 1 .txt @ end.
#         fullpath = os.path.join(INSTRUMENT_FILE_DIRECTORY, self.instrument_filename)
#         instrument_dict = dict_from_directives_file(fullpath)
#         instrument_dict_directives = set(instrument_dict.keys())
#         required_directives = set(REQUIRED_INSTRUMENT_DIRECTIVES)
#         missing_directives = required_directives - instrument_dict_directives
#         if len(missing_directives) > 0:
#             for md in missing_directives:
#                 print(' >>>>> ERROR: instrument file is missing directive', md + '.')
#             exit(0)
#
#         # Verify no keys overlap, then combine dicts:
#         directives_in_both_dicts = defaults_dict_directives.intersection(instrument_dict_directives)
#         if len(directives_in_both_dicts) > 0:
#             for db in directives_in_both_dicts:
#                 print(' >>>>> WARNING: directive', db, 'appears in both directive and instrument files.')
#         self._data = {**defaults_dict, **instrument_dict}  # nb: instrument_dict overrides defaults_dict.
#
#         # Cast to floats those values that actually represent floats:
#         for key in ['MP_RI_COLOR', 'MIN_CATALOG_R_MAG', 'MAX_CATALOG_R_MAG',
#                     'MAX_CATALOG_DR_MMAG', 'MIN_CATALOG_RI_COLOR', 'MAX_CATALOG_RI_COLOR',
#                     'PIXEL_SHIFT_TOLERANCE', 'FWHM_NOMINAL', 'CCD_GAIN', 'ADU_SATURATION',
#                     'PINPOINT_PIXEL_SCALE_FACTOR']:
#             self._data[key] = float(self._data[key])
#
#         # Split long transform strings into value lists:
#         self._data['FIT_TRANSFORM'] = self._data['FIT_TRANSFORM'].split()
#         if not isinstance(self._data['TRANSFORM'], list):
#             self._data['TRANSFORM'] = [self._data['TRANSFORM']]
#         self._data['TRANSFORM'] = [tr.split() for tr in self._data['TRANSFORM']]
#
#     def __str__(self):
#         return 'Settings object from instrument file ' + self.instrument_filename
#
#     # Allow direct access as settings=Settings(); value = settings['somekey'].
#     def __getitem__(self, key):
#         return self._data.get(key, None)  # return None if key absent.
#
#     def get(self, key, default_value):
#         return self._data.get(key, default_value)
#
#
# class Control:
#     """ Holds data from control.txt file. Assumes current working directory is set by start().
#         Supplied by user in text file 'control.txt' within session directory.
#         Modified by user to control actual data reduction, including comp start selection
#             and the fit model, including terms to include, MP color, and transform type.
#         Expected items:
#         FIT_EXTINCTION: True if user includes extinction in comp-star regression model. [boolean]
#         FIT_JD: True if user includes linear time term (drift) in comp-star regression model. [boolean]
#         FIT_TRANSFORM: True if user includes transform in comp-star regression model. [boolean]
#         FIT_VIGNETTE: True if user includes parabolic vignetting in comp-star regression model. [boolean]
#         FIT_XY: True if user includes linear X- and Y-gradients in comp-star regression model. [boolean]
#         MAX_CATALOG_DR_MMAG: session's maximum catalog comp-star mag uncertainty in Sloan r [float]
#         MAX_CATALOG_R_MAG: session's maximum catalog comp-star mag in Sloan r [float]
#         MIN_CATALOG_R_MAG: session's maximum catalog comp-star mag in Sloan r [float]
#         MAX_CATALOG_RI_COLOR: session's default maximum catalog comp-star color in Sloan r-i [float]
#         MIN_CATALOG_RI_COLOR: session's default minimum catalog comp-star color in Sloan r-i [float]
#         MP_RI_COLOR: user's session MP color in Sloan r-i, e.g. +0.22 [float]
#         MP_LOCATION: 2-list of MP location specifications, from an early and a late image of the session,
#             e.g., [['MP_191-0001-Clear.fts', 826.4, 1077.4], ['MP_191-0028-Clear.fts', 1144.3, 1099.3]]
#         REF_STAR_LOCATION: list of at least 2 ref star location specifications, usually in the same image,
#             e.g., [['MP_191-0001-Clear.fts', 790.6, 1115.0],
#                    ['MP_191-0001-Clear.fts', 819.3, 1011.7],
#                    ['MP_191-0001-Clear.fts', 1060.4, 1066.0]]
#         IS_VALID: True if file 'control.txt' parsed without errors. [boolean]
#         ERRORS: Errors encountered while parsing file 'control.txt'; typically []. [list of strings]
#     """
#     def __init__(self):
#         context = get_context()
#         if context is None:
#             return
#         this_directory, mp_string, an_string, filter_string = context
#         self.fullpath = os.path.join(this_directory, CONTROL_FILENAME)
#         control_dict = dict_from_directives_file(self.fullpath)
#         control_directives = set(control_dict.keys())
#         required_directives = set(REQUIRED_CONTROL_DIRECTIVES)
#         missing_directives = required_directives - control_directives
#         control_dict['IS_VALID'] = True
#         control_dict['ERRORS'] = []
#         if len(missing_directives) > 0:
#             for md in missing_directives:
#                 print(' >>>>> ERROR: control file is missing directive', md + '.')
#                 control_dict['IS_VALID'] = False
#                 control_dict['ERRORS'].append('Missing directive: ' + md)
#
#         # Verify at least 2 REF_STAR_LOCATION and exactly 2 MP_LOCATION entries:
#         rsl = control_dict['REF_STAR_LOCATION']
#         if not len(rsl) >= 2:
#             print(' >>>>> ERROR: only', str(len(rsl)), 'REF_STAR_LOCATION lines, but >= 2 required.')
#             control_dict['IS_VALID'] = False
#             control_dict['ERRORS'].append('REF_STAR_LOCATION count: ' + str(len(rsl)) + ' but >=2 required.')
#         mpl = control_dict['MP_LOCATION']
#         if not len(mpl) == 2:
#             print(' >>>>> ERROR:', str(len(mpl)), 'MP_LOCATION lines, but exactly 2 required.')
#             control_dict['IS_VALID'] = False
#             control_dict['ERRORS'].append('MP_LOCATION count: ' + str(len(mpl)) +
#                                           ' but exactly 2 required.')
#
#         # Cast values into proper types:
#         for key in ['MP_RI_COLOR', 'MIN_CATALOG_R_MAG', 'MAX_CATALOG_R_MAG',
#                     'MAX_CATALOG_DR_MMAG', 'MIN_CATALOG_RI_COLOR', 'MAX_CATALOG_RI_COLOR']:
#             try:
#                 control_dict[key] = float(control_dict[key])
#             except ValueError:
#                 print(' >>>>> ERROR: non-numeric value ' + str(control_dict[key]) +
#                       'for directive ' + key + '.')
#                 control_dict['IS_VALID'] = False
#                 control_dict['ERRORS'].append('non-numeric value ' + str(control_dict[key]) +
#                                               'for directive ' + key + '.')
#         for key in ['FIT_EXTINCTION', 'FIT_VIGNETTE', 'FIT_XY', 'FIT_JD']:
#             control_dict[key] = True if control_dict[key].upper()[0] == 'Y' else False
#         new_values = []
#         for raw_string in control_dict['REF_STAR_LOCATION']:
#             tokens = raw_string.strip().rsplit(maxsplit=2)
#             if len(tokens) != 3:
#                 print(' >>>>> ERROR: bad syntax in REF_STAR_LOCATION entry ' + raw_string)
#                 control_dict['IS_VALID'] = False
#                 control_dict['ERRORS'].append('bad syntax in REF_STAR_LOCATION entry ' + raw_string)
#             try:
#                 new_value = [tokens[0], float(tokens[1]), float(tokens[2])]
#             except ValueError:
#                 print(' >>>>> ERROR: non-numeric in REF_STAR_LOCATION entry ' + raw_string)
#                 control_dict['IS_VALID'] = False
#                 control_dict['ERRORS'].append('non-numeric in REF_STAR_LOCATION entry ' + raw_string)
#                 new_value = [tokens[0], None, None]
#             new_values.append(new_value)
#         control_dict['REF_STAR_LOCATION'] = new_values
#         new_values = []
#         for raw_string in control_dict['MP_LOCATION']:
#             tokens = raw_string.strip().rsplit(maxsplit=2)
#             if len(tokens) != 3:
#                 print(' >>>>> ERROR: bad syntax in MP_LOCATION entry ' + raw_string)
#                 control_dict['IS_VALID'] = False
#                 control_dict['ERRORS'].append('bad syntax in MP_LOCATION entry ' + raw_string)
#             try:
#                 new_value = [tokens[0], float(tokens[1]), float(tokens[2])]
#             except ValueError:
#                 print(' >>>>> ERROR: non-numeric in MP_LOCATION entry ' + raw_string)
#                 control_dict['IS_VALID'] = False
#                 control_dict['ERRORS'].append('non-numeric in MP_LOCATION entry ' + raw_string)
#                 new_value = [tokens[0], None, None]
#             new_values.append(new_value)
#         control_dict['MP_LOCATION'] = new_values
#         self._data = control_dict
#
#     def __str__(self):
#         return 'Control object from ' + self.fullpath
#
#     # Allow direct access as control=Control(); value = control['somekey'].
#     def __getitem__(self, key):
#         return self._data.get(key, None)  # return None if key absent.

# def test_dict_from_directives_file():
#     d = util.dict_from_directives_file(DEFAULTS_FULLPATH)
#     assert isinstance(d, dict)
#     all_keys = set(d.keys())
#     required_keys = set(workflow_session.REQUIRED_DEFAULT_DIRECTIVES)
#     assert len(required_keys - all_keys) == 0  # all required must be present.
#     assert d['INSTRUMENT'] == 'Borea'
#     assert d['MAX_CATALOG_R_MAG'] == '16'
#     assert d['FIT_JD'] == 'Yes'
#     assert 'INVALID_KEY' not in all_keys  # absent key returns None.