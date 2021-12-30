import pdb
import os
from configparser import ConfigParser

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