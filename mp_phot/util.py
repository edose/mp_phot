__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import numpy as np
import pandas as pd

VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']


def get_mp_filenames(directory):
    all_filenames = pd.Series([e.name for e in os.scandir(directory) if e.is_file()])
    extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in all_filenames])
    is_fits = [ext.lower() in VALID_FITS_FILE_EXTENSIONS for ext in extensions]
    fits_filenames = all_filenames[is_fits]
    mp_filenames = [fn for fn in fits_filenames if fn.startswith('MP_')]
    return mp_filenames


def dict_from_directives_file(fullpath):
    """ For each directive line in file, return a value [string] or list of values [list of strings].
        Returned dict includes ONLY strings exactly as given without context, e.g., '16' for 16 and 'Yes'.
        It is up to the calling routine to cast into proper types and/or interpret these string values.
    :param fullpath: [string]
    :return: dict of key=directive [string], value=value [string] or list of values [list of strings].
    """
    with open(fullpath) as f:
        lines = f.readlines()
    lines = [line for line in lines if line is not None]  # remove empty list elements
    lines = [line.split(";")[0] for line in lines]  # remove all comments
    lines = [line.strip() for line in lines]  # remove lead/trail blanks
    lines = [line for line in lines if line != '']  # remove empty lines
    lines = [line for line in lines if line.startswith('#')]  # keep only directive lines
    data_dict = dict()
    for line in lines:
        splitline = line.split(maxsplit=1)
        if len(splitline) != 2:
            print(' >>>>> ERROR: File', fullpath, 'cannot parse line', line)
        else:
            directive = (splitline[0])[1:].strip().upper()
            value = splitline[1].strip()
            previous_value = data_dict.get(directive, None)
            if previous_value is None:
                data_dict[directive] = value  #
            else:
                if not isinstance(previous_value, list):
                    data_dict[directive] = [data_dict[directive]]  # ensure previous value is a list.
                data_dict[directive].append(value)
    return data_dict


