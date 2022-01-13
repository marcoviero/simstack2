import pdb
import os
import json
import numpy as np
import pandas as pd

class Skycatalogs:
	def __init__(self, config_dict):

		self.config_dict = config_dict

	def import_catalog(self):

		self.catalog_dict = {}

		catalog_params = self.config_dict['catalog']
		path_catalog = os.path.join(self.parse_path(catalog_params['path']), catalog_params['file'])
		if os.path.isfile(path_catalog):
			self.catalog_dict['table'] = pd.read_table(path_catalog, sep=',')
		else:
			print("Catalog not found: "+path_catalog)
			pdb.set_trace()

		self.split_table_into_populations()

	def split_table_into_populations(self):

		# Make new table starting with RA and DEC
		astrometry_keys = json.loads(self.config_dict['catalog']['astrometry'])
		self.split_table = {}
		self.split_table['table'] = pd.DataFrame(self.catalog_dict['table'][astrometry_keys.values()])
		self.split_table['table'].rename(columns={astrometry_keys["ra"]: "ra", astrometry_keys["dec"]: "dec"}, inplace=True)

		# Split catalog by classification type
		split_dict = json.loads(self.config_dict['catalog']['classification'])
		split_type = split_dict.pop('split_type')

		# By labels means unique values inside columns (e.g., "CLASS" = [0,1,2])
		if 'labels' in split_type:
			self.separate_by_label(split_dict, self.catalog_dict['table'])

		# By uvj means it splits into star-forming and quiescent galaxies via the u-v/v-j method.
		if 'uvj' in split_type:
			self.separate_sf_qt(split_dict, self.catalog_dict['table'])

	def separate_by_label(self, split_dict, table, add_background=False):
		#table = self.catalog_dict['table']
		self.parameter_names = {}
		label_keys = list(split_dict.keys())
		for key in label_keys:
			if type(split_dict[key]['bins']) is str:
				bins = json.loads(split_dict[key]['bins'])
				self.parameter_names[key] = ["_".join([key, str(bins[i]), str(bins[i + 1])]) for i in range(len(bins[:-1]))]
			elif type(split_dict[key]['bins']) is dict:
				bins = len(split_dict[key]['bins'])
				self.parameter_names[key] = ["_".join([key, str(i)]) for i in range(bins)]
			else:
				bins = split_dict[key]['bins']
				self.parameter_names[key] = ["_".join([key, str(i)]) for i in range(bins)]
			# Categorize using pandas.cut.  So good.
			col = pd.cut(table[split_dict[key]['id']], bins=bins, labels=False)
			col.name = key  # Rename column to label
			# Add column to table
			self.split_table['table'] = self.split_table['table'].join(col)

		# Name Cube Layers (i.e., parameters)
		self.split_table['parameter_labels'] = []
		for ipar in self.parameter_names[label_keys[0]]:
			for jpar in self.parameter_names[label_keys[1]]:
				if len(label_keys) > 2:
					for kpar in self.parameter_names[label_keys[2]]:
						if len(label_keys) > 3:
							for lpar in self.parameter_names[label_keys[3]]:
								pn = "__".join([ipar, jpar, kpar, lpar])
								self.split_table['parameter_labels'].append(pn)
						else:
							pn = "__".join([ipar, jpar, kpar])
							self.split_table['parameter_labels'].append(pn)
				else:
					pn = "__".join([ipar, jpar])
					self.split_table['parameter_labels'].append(pn)
		if add_background:
			self.split_table['parameter_labels'].append('background_layer')
		#pdb.set_trace()

	def separate_sf_qt(self, split_dict, table):
		#table = self.catalog_dict['table']

		uvkey = split_dict['uvj']["bins"]['U-V']
		vjkey = split_dict['uvj']["bins"]['V-J']
		zkey = split_dict['redshift']["id"]

		# Find quiescent galaxies using UVJ criteria
		ind_zlt1 = (table[uvkey] > 1.3) & (table[vjkey] < 1.5) & (table[zkey] < 1) & \
				   (table[uvkey] > (table[vjkey] * 0.88 + 0.69))
		ind_zgt1 = (table[uvkey] > 1.3) & (table[vjkey] < 1.5) & (table[zkey] >= 1) & \
				   (table[uvkey] > (table[vjkey] * 0.88 + 0.59))

		# Add sfg column
		sfg = np.ones(len(table))
		sfg[ind_zlt1] = 0
		sfg[ind_zgt1] = 0
		class_label = split_dict['uvj']["id"]  # typically 'sfg', but can be anything.
		table[class_label] = sfg

		#pdb.set_trace()
		self.separate_by_label(split_dict, table)
