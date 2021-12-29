import pdb
import os
import sys
import numpy as np
import gc
from astropy.wcs import WCS
from configparser import ConfigParser
from utils import circle_mask
from utils import clean_args
from utils import clean_nans
from utils import gauss_kern
from utils import smooth_psf
from lmfit import Parameters, minimize, fit_report
from simstackalgorithm import SimstackAlgorithm

pi = 3.141592653589793
L_sun = 3.839e26  # W
c = 299792458.0  # m/s
conv_sfr = 1.728e-10 / 10 ** (.23)
conv_luv_to_sfr = 2.17e-10
conv_lir_to_sfr = 1.72e-10
a_nu_flux_to_mass = 6.7e19
flux_to_specific_luminosity = 1.78  # 1e-23 #1.78e-13
h = 6.62607004e-34  # m2 kg / s  #4.13e-15 #eV/s
k = 1.38064852e-23  # m2 kg s-2 K-1 8.617e-5 #eV/K

class SimstackSettings:
    config_dict = {}

    def __init__(self, param_file_path):

        self.config_dict = self.get_params_dict(param_file_path)

    def parse_path(self, path_in):
        path_in = path_in.split(" ")
        if len(path_in) == 1:
            return path_in[0]
        else:
            return os.path.join(os.environ[path_in[0]], path_in[1])

    def get_params_dict(self, param_file_path):
        config = ConfigParser()
        config.read(param_file_path)

        dict_out = {}
        for section in config.sections():
            dict_sect = {}
            for (each_key, each_val) in config.items(section):
                dict_sect[each_key] = each_val.replace("'", '"')

            dict_out[section] = dict_sect

        return dict_out

    def write_config_file(self, params_out, config_filename_out):
        config_out = ConfigParser()

        for ikey, idict in params_out.items():
            if not config_out.has_section(ikey):

                config_out.add_section(ikey)
                for isect, ivals in idict.items():
                    # pdb.set_trace()
                    # print(ikey, isect, ivals)
                    config_out.set(ikey, isect, str(ivals))

        # Write config_filename_out (check if overwriting externally)
        with open(config_filename_out, 'w') as conf:
            config_out.write(conf)

class SimstackStacking(SimstackSettings, SimstackAlgorithm):

    def __init__(self, param_path_file, read_maps=True, read_catalog=True):
        super().__init__(param_path_file)

        if read_catalog:
            self.import_catalog(keep_raw_table=True)

        if read_maps:
            self.import_maps()

        self.perform_simstack()
