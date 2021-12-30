from simstackalgorithm import SimstackAlgorithm

class SimstackWrapper(SimstackAlgorithm):

    def __init__(self, param_path_file, read_maps=True, read_catalog=True, stack_automatically=False):
        super().__init__(param_path_file)

        if read_catalog:
            self.import_catalog()

        if read_maps:
            self.import_maps()

        if stack_automatically:
            self.perform_simstack()
