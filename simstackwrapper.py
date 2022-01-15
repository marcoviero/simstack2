import pdb
import os

from simstackalgorithm import SimstackAlgorithm

class SimstackWrapper(SimstackAlgorithm):
    ''' SimstackWrapper consolidates each step required to stack:
        - Read in parameters from config.ini file
        - Read in catalogs
        - Read in maps
        - Split catalog into bins specified in config file
        - Run stacking algorithm, which includes:
            -- create convolved layer cube at each wavelength [and optionally redshift]
        - Parse results into user-friendly pandas DataFrames

        :param param_path_file: (str)  path to the config.ini file
        :param read_maps: (bool) [optional; default True] If prefer to do this manually then set False
        :param read_catalogs: (bool) [optional; default True] If prefer to do this manually then set False
        :param stack_automatically: (bool) [optional; default False] If prefer to do this automatically then set True

        TODO:
        - options: 1 read, 2 overwrite, 3 write new
        - counts in bins
        - restructure dicts
        - agn selection to pops
        - optimal binning (bayesian blocks?)
        - lookback time
        - CIB estimates
        - CUBE OF BEST FITS

    '''
    def __init__(self, param_file_path, read_maps=False, read_catalog=False, stack_automatically=False):
        super().__init__(param_file_path)

        if 'shortname' in self.config_dict['io']:
            shortname = self.config_dict['io']['shortname']
        else:
            shortname = os.path.basename(param_file_path).split('.')[0]

        out_file_path = os.path.join(self.parse_path(self.config_dict['io']['output_folder']), shortname)

        continue_stack = True
        force_stack = True
        if os.path.isdir(out_file_path) and force_stack is False:
            if 'import_results' in self.config_dict['io']:
                if self.config_dict['io']['import_results'] == 'True':
                    import_pickle_path = os.path.join(out_file_path, shortname)+'.pkl'
                    imported_object = self.import_saved_pickles(import_pickle_path)
                    print('Importing Saved Results', os.path.basename(import_pickle_path))
                    self.replace_self(imported_object)
                    continue_stack = False
            else:
                value = input("Enter 1 to Import Results; 2 to Overwrite Results; 3 to Save New Results :\n")
                print(f'You entered {value}')
                if value == 1:
                    import_pickle_path = os.path.join(out_file_path, shortname)+'.pkl'
                    imported_object = self.import_saved_pickles(import_pickle_path)
                    print('Importing Saved Results', os.path.basename(import_pickle_path))
                    self.replace_self(imported_object)
                    continue_stack = False
                elif value == 2:
                    self.config_dict['io']['overwrite_results'] = 'True'
                    print('Overwriting Saved Results')
                    continue_stack = True

        #pdb.set_trace()

        if continue_stack:
            if read_catalog:
                self.import_catalog()  # This happens in skycatalogs

            if read_maps:
                self.import_maps()  # This happens in skymaps

            if stack_automatically:
                self.perform_simstack()  # This happens in simstackalgorithm

            if self.stack_successful:
                self.parse_results()  # This happens in simstackresults
