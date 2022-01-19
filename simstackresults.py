import pdb
import os
import json
import numpy as np
import pandas as pd

class SimstackResults():

	results_dict = {}

	def __init__(self):
		super().__init__()

	def parse_results(self, beta_rj=1.8):

		wavelength_keys = list(self.maps_dict.keys())
		wavelengths = []
		split_dict = json.loads(self.config_dict['catalog']['classification'])
		split_type = split_dict.pop('split_type')
		label_keys = list(split_dict.keys())
		label_dict = self.config_dict['parameter_names']
		ds = [len(label_dict[k]) for k in label_dict]

		sed_flux_array = np.zeros([len(wavelength_keys), *ds])
		sed_error_array = np.zeros([len(wavelength_keys), *ds])
		for k, key in enumerate(wavelength_keys):
			self.results_dict[key] = {}
			self.results_dict[key]['wavelength'] = self.maps_dict[key]['wavelength']
			wavelengths.append(self.maps_dict[key]['wavelength'])

			flux_array = np.zeros(ds)
			error_array = np.zeros(ds)
			results_object = self.maps_dict[key]['stacked_flux_densities']

			for z, zval in enumerate(self.config_dict['catalog']['distance_labels']):
				if 'all_redshifts' in results_object:
					zlab = 'all_redshifts'
				else:
					zlab = zval
				for i, ival in enumerate(label_dict[label_keys[1]]):
					if len(label_keys) > 2:
						for j, jval in enumerate(label_dict[label_keys[2]]):
							label = "__".join([zval, ival, jval]).replace('.', 'p')
							#print(label)
							# CHECK THAT LABEL EXISTS FIRST
							if label in results_object[zlab].params:
								flux_array[z, i, j] = results_object[zlab].params[label].value
								error_array[z, i, j] = results_object[zlab].params[label].stderr
							#else:
								#print(label, ' does not exist')
					else:
						label = "__".join([zval, ival]).replace('.', 'p')
						#print(label)
						if label in results_object[zlab].params:
							flux_array[z, i] = results_object[zlab].params[label].value
							error_array[z, i] = results_object[zlab].params[label].stderr
						#else:
							#print(label, ' does not exist')

			z_bins = [i.replace('p', '.').split('_')[1:] for i in self.config_dict['catalog']['distance_labels']]
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
				if len(label_keys) > 2:
					for j, jval in enumerate(label_dict[label_keys[2]]):
						z_dict['flux_density'][zval][jval] = flux_array[z, :, j]
						z_dict['std_error'][zval][jval] = error_array[z, :, j]
				else:
					z_dict['flux_density'][zval] = flux_array[z, :]
					z_dict['std_error'][zval] = error_array[z, :]

			m_dict = {'flux_density': {}, 'std_error': {}, 'stellar_mass': []}
			for i, ival in enumerate(label_dict[label_keys[1]]):
				m_dict['flux_density'][ival] = {}
				m_dict['std_error'][ival] = {}
				m_dict['stellar_mass'].append(ival)
				if len(label_keys) > 2:
					for j, jval in enumerate(label_dict[label_keys[2]]):
						m_dict['flux_density'][ival][jval] = flux_array[:, i, j]
						m_dict['std_error'][ival][jval] = error_array[:, i, j]
				else:
					m_dict['flux_density'][ival] = flux_array[:, i]
					m_dict['std_error'][ival] = error_array[:, i]

			self.results_dict[key][label_keys[0]] = z_dict
			self.results_dict[key][label_keys[1]] = m_dict

			sed_flux_array[k, :, :] = flux_array
			sed_error_array[k, :, :] = error_array

		# Store Wavelengths
		#self.results_dict['wavelengths'] = wavelengths

		# Organize into SEDs
		self.results_dict['SED_df'] = {'flux_density': {}, 'std_error': {}, 'SED': {}, 'LIR': {},
									   'wavelengths': wavelengths}
		for z, zlab in enumerate(label_dict[label_keys[0]]):
			if len(label_keys) > 2:
				for j, jlab in enumerate(label_dict[label_keys[2]]):
					if zlab not in self.results_dict['SED_df']['flux_density']:
						self.results_dict['SED_df']['flux_density'][zlab] = {}
						self.results_dict['SED_df']['std_error'][zlab] = {}
						self.results_dict['SED_df']['LIR'][zlab] = {}
						self.results_dict['SED_df']['SED'][zlab] = {}

					self.results_dict['SED_df']['flux_density'][zlab][jlab] = \
						pd.DataFrame(sed_flux_array[:, z, :, j], index=wavelengths, columns=label_dict[label_keys[1]])
					self.results_dict['SED_df']['std_error'][zlab][jlab] = \
						pd.DataFrame(sed_error_array[:, z, :, j], index=wavelengths, columns=label_dict[label_keys[1]])

					self.results_dict['SED_df']['LIR'][zlab][jlab] = {}
					self.results_dict['SED_df']['SED'][zlab][jlab] = {}
					for i, ilab in enumerate(label_dict[label_keys[1]]):
						tst_m = self.fast_sed_fitter(wavelengths, sed_flux_array[:, z, i, j], sed_error_array[:, z, i, j],
													 betain=1.8)
						tst_LIR = self.fast_Lir(tst_m, z_mid[z])
						self.results_dict['SED_df']['LIR'][zlab][jlab][ilab] = tst_LIR.value
						self.results_dict['SED_df']['SED'][zlab][jlab][ilab] = tst_m
			else:
				self.results_dict['SED_df']['flux_density'][zlab] = \
					pd.DataFrame(sed_flux_array[:, z, :], index=wavelengths, columns=label_dict[label_keys[1]])
				self.results_dict['SED_df']['std_error'][zlab] = \
					pd.DataFrame(sed_error_array[:, z, :], index=wavelengths, columns=label_dict[label_keys[1]])

				for i, ilab in enumerate(label_dict[label_keys[1]]):
					tst_m = self.fast_sed_fitter(wavelengths, sed_flux_array[:, z, i], sed_error_array[:, z, i],
												 betain=1.8)
					tst_LIR = self.fast_Lir(tst_m, z_mid[z])

					if zlab not in self.results_dict['SED_df']['LIR']:
						self.results_dict['SED_df']['LIR'][zlab] = {}
						self.results_dict['SED_df']['SED'][zlab] = {}

					self.results_dict['SED_df']['LIR'][zlab][ilab] = tst_LIR.value
					self.results_dict['SED_df']['SED'][zlab][ilab] = tst_m

		#pdb.set_trace()


