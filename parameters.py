import pdb
import numpy as np
from configparser import ConfigParser
import logging

"""
Functions to read/write parameters from/to .ini files using ConfigParser
"""


def get_params_dict(param_file_path):
    config = ConfigParser()
    config.read(param_file_path)

    dict_out = {}
    dict_raw = {}
    for section in config.sections():
        dict_sect = {}
        for (each_key, each_val) in config.items(section):
            dict_sect[each_key] = each_val
            dict_raw[each_key] = each_val

        dict_out[section] = dict_sect

    return dict_out


def write_config_file(params_out, config_filename_out):
    config_out = ConfigParser()

    for ikey, idict in params_out.items():
        if not config_out.has_section(ikey):

            config_out.add_section(ikey)
            for isect, ivals in idict.items():
                # pdb.set_trace()
                # print(ikey, isect, ivals)
                config_out.set(ikey, isect, str(ivals))

    # pdb.set_trace()
    # Write config_filename_out (check if overwriting externally)

    with open(config_filename_out, 'w') as conf:
        config_out.write(conf)

### FOR TESTING ###
if __name__=='__main__':
    import os, sys
    import pprint

    param_fp = sys.argv[1]
    print("")
    print("Testing %s on %s..." % (os.path.basename(__file__), param_fp))
    print("")
    pprint.pprint(get_params_dict(param_fp))
