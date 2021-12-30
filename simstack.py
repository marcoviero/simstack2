from simstackalgorithm import SimstackAlgorithm

class SimstackWrapper(SimstackAlgorithm):

    def __init__(self, param_path_file, read_maps=True, read_catalog=True):
        super().__init__(param_path_file)

        if read_catalog:
            self.import_catalog(keep_raw_table=True)

        if read_maps:
            self.import_maps()

        self.perform_simstack()
