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
    '''
    def __init__(self, param_path_file, read_maps=False, read_catalog=False, stack_automatically=False):
        super().__init__(param_path_file)

        if read_catalog:
            self.import_catalog()  # This happens in skycatalogs

        if read_maps:
            self.import_maps()  # This happens in skymaps

        if stack_automatically:
            self.perform_simstack()  # This happens in simstackalgorithm

        if self.stack_successful:
            self.parse_results()  # This happens in simstackresults
