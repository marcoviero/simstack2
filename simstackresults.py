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
		wavelengths = []
		split_dict = json.loads(self.config_dict['catalog']['classification'])
		split_type = split_dict.pop('split_type')
		label_keys = list(split_dict.keys())
		label_dict = self.parameter_names
		ds = [len(label_dict[k]) for k in label_dict]
		#pdb.set_trace()
		#sed_array = {'flux_density': np.zeros([len(wavelength_keys), *ds]),
		#			 'uncertainty': np.zeros([len(wavelength_keys), *ds]) }

		for k, key in enumerate(wavelength_keys):
			self.results_dict[key] = {}
			self.results_dict[key]['wavelength'] = self.maps_dict[key]['wavelength']
			wavelengths.append(self.maps_dict[key]['wavelength'])

			flux_array = np.zeros(ds)
			error_array = np.zeros(ds)
			results_object = self.maps_dict[key]['stacked_flux_densities']

			z_label = []
			for z, zval in enumerate(results_object):
				z_label.append(zval)
				for i, ival in enumerate(label_dict[label_keys[1]]):
					if len(label_keys) > 2:
						for j, jval in enumerate(label_dict[label_keys[2]]):
							label = "__".join([zval, ival, jval]).replace('.', 'p')
							#print(label)
							flux_array[z, i, j] = results_object[zval].params[label].value
							error_array[z, i, j] = results_object[zval].params[label].stderr
							#sed_array['flux_density'][k, z, i, j] = flux_array[z, i, j]
							#pdb.set_trace()
					else:
						label = "__".join([zval, ival]).replace('.', 'p')
						#print(label)
						flux_array[z, i] = results_object[zval].params[label].value
						error_array[z, i] = results_object[zval].params[label].stderr

			z_bins = [i.replace('p', '.').split('_')[1:] for i in z_label]
			z_mid = [(float(i[0]) + float(i[1]))/2 for i in z_bins]

			self.results_dict[key]['results_df'] = {}
			if len(label_keys) > 2:
				self.results_dict[key]['results_df']['flux_df'] = {}
				self.results_dict[key]['results_df']['error_df'] = {}
				for j, jval in enumerate(label_dict[label_keys[2]]):
					self.results_dict[key]['results_df']['flux_df'][jval] = \
						pd.DataFrame(flux_array[:, :, j],
									 columns=[label_dict[label_keys[1]]], index=z_mid) # , index = [label_dict[label_keys[0]]])
					self.results_dict[key]['results_df']['error_df'][jval] = \
						pd.DataFrame(error_array[:, :, j],
									 columns=[label_dict[label_keys[1]]], index=z_mid)
			else:
				self.results_dict[key]['results_df']['flux_df'] = \
					pd.DataFrame(flux_array[:, :],
								 columns=[label_dict[label_keys[1]]], index=z_mid)
				self.results_dict[key]['results_df']['error_df'] = \
					pd.DataFrame(error_array[:, :],
								 columns=[label_dict[label_keys[1]]], index=z_mid)

			z_dict = {'flux_density': {}, 'std_error': {}, 'redshift': []}
			for z, zval in enumerate(results_object):
				z_dict['flux_density'][zval] = {}
				z_dict['std_error'][zval] = {}
				z_dict['redshift'].append(zval)
				#z_dict[zval] = {}
				if len(label_keys) > 2:
					for j, jval in enumerate(label_dict[label_keys[2]]):
						z_dict['flux_density'][zval][jval] = flux_array[z, :, j]
						z_dict['std_error'][zval][jval] = error_array[z, :, j]
						#z_dict[zval][jval] = {"flux_density": flux_array[z, :, j], "std_error": error_array[z, :, j]}
				else:
					z_dict['flux_density'][zval] = flux_array[z, :]
					z_dict['std_error'][zval] = error_array[z, :]
					#z_dict[zval] = {"flux_density": flux_array[z, :], "std_error": error_array[z, :]}

			m_dict = {'flux_density': {}, 'std_error': {}, 'stellar_mass': []}
			#m_dict = {}
			for i, ival in enumerate(label_dict[label_keys[1]]):
				m_dict['flux_density'][ival] = {}
				m_dict['std_error'][ival] = {}
				m_dict['stellar_mass'].append(ival)
				#m_dict[ival] = {}
				if len(label_keys) > 2:
					for j, jval in enumerate(label_dict[label_keys[2]]):
						m_dict['flux_density'][ival][jval] = flux_array[:, i, j]
						m_dict['std_error'][ival][jval] = error_array[:, i, j]
						#m_dict[ival][jval] = {"flux_density": flux_array[:, i, j], "std_error": error_array[:, i, j]}
				else:
					m_dict['flux_density'][ival] = flux_array[:, i]
					m_dict['std_error'][ival] = error_array[:, i]
					#m_dict[ival] = {"flux_density": flux_array[:, i], "std_error": error_array[:, i]}

			self.results_dict[key][label_keys[0]] = z_dict
			self.results_dict[key][label_keys[1]] = m_dict

		self.results_dict['wavelengths'] = wavelengths


		#pdb.set_trace()


