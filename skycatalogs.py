import pdb
import os
import json
import numpy as np
import pandas as pd

class Skycatalogs:
	def __init__(self, config_dict):

		self.config_dict = config_dict

	def import_catalog(self, keep_raw_table=False):

		self.catalog_dict = {}

		catalog_params = self.config_dict['catalog']
		path_catalog = os.path.join(self.parse_path(catalog_params['path']), catalog_params['file'])
		if os.path.isfile(path_catalog):
			table = pd.read_table(path_catalog, sep=',')
			if keep_raw_table:
				self.catalog_dict['table'] = table
		else:
			print("Catalog not found: "+path_catalog)

		split_dict = json.loads(self.config_dict['catalog']['classification'])
		split_type = split_dict.pop('split_type')

		self.split_table_by_populations(table, split_dict, split_type)

	def split_table_by_populations(self, table, split_dict, split_type='uvj'):

		#table = self.catalog_dict['table']
		astrometry_keys = json.loads(self.config_dict['catalog']['astrometry'])
		self.split_table = pd.DataFrame(table[astrometry_keys.values()])
		self.split_table.rename(columns={astrometry_keys["ra"]: "ra", astrometry_keys["dec"]: "dec"}, inplace=True)

		if 'uvj' in split_type:
			self.separate_sf_qt()

		if 'labels' in split_type:
			split_keys = list(split_dict.keys())
			for key in split_keys:
				if type(split_dict[key]['bins']) is str:
					bins = json.loads(split_dict[key]['bins'])
					parameter_names = ["_".join([key, str(bins[i]), str(bins[i + 1])]) for i in range(len(bins[:-1]))]
				else:
					bins = split_dict[key]['bins']
					parameter_names = ["_".join([key, str(i)]) for i in range(bins)]
				labels = False
				col = pd.cut(table[split_dict[key]['id']], bins=bins, labels=labels)
				col.name = key
				self.split_table = self.split_table.join(col)

	def separate_sf_qt(self):
		table = self.catalog_dict['table']
		nsrc = len(table)
		sfg = np.ones(nsrc)

		zkey = json.loads(self.config_dict['catalog']['classification'])['redshift']["id"]
		zbins = json.loads(json.loads(self.config_dict['catalog']['classification'])['redshift']["bins"])
		mkey = json.loads(self.config_dict['catalog']['classification'])['stellar_mass']["id"]
		mbins = json.loads(json.loads(self.config_dict['catalog']['classification'])['stellar_mass']["bins"])

		#uvkey = self.config_dict['general']['classification']
		pdb.set_trace()
		for i in range(nsrc):
			if (self.table[self.uvkey][i] > 1.3) and (self.table[self.vjkey][i] < 1.5):
				if (self.table[self.zkey][i] < 1):
					if (self.table[self.uvkey][i] > (self.table[self.vjkey][i] * 0.88 + 0.69)): sfg[i] = 0
				if (self.table[self.zkey][i] > 1):
					if (self.table[self.uvkey][i] > (self.table[self.vjkey][i] * 0.88 + 0.59)): sfg[i] = 0
		return sfg
