import pdb
import os
import json
import numpy as np
import pandas as pd

class SimstackResults:

	results_dict = {}

	def __init__(self):
		pass

	def parse_results(self):

		wavelength_keys = list(self.maps_dict.keys())
		split_dict = json.loads(self.config_dict['catalog']['classification'])
		split_type = split_dict.pop('split_type')
		label_keys = list(split_dict.keys())
		label_dict = self.parameter_names
		ds = [len(label_dict[k]) for k in label_dict]

		for k, key in enumerate(wavelength_keys):
			self.results_dict[key] = {}
			flux_array = np.zeros(ds)
			error_array = np.zeros(ds)
			results_object = self.maps_dict[key]['stacked_flux_densities']
			for z, zval in enumerate(results_object):
				#layer_keys = list(results_object[z].params.keys())
				for i, ival in enumerate(label_dict[label_keys[1]]):
					if len(label_keys) > 2:
						for j, jval in enumerate(label_dict[label_keys[2]]):
							label = "__".join([zval, ival, jval]).replace('.', 'p')
							#print(label)
							flux_array[z, i, j] = results_object[zval].params[label].value
							error_array[z, i, j] = results_object[zval].params[label].stderr
					else:
						label = "__".join([zval, ival]).replace('.', 'p')
						flux_array[z, i] = results_object[zval].params[label].value
						error_array[z, i] = results_object[zval].params[label].stderr

			self.results_dict[key]['flux_density'] = flux_array
			self.results_dict[key]['std_error'] = error_array

		#pdb.set_trace()


