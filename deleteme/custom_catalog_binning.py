import pdb
import six
import numpy as np
from astropy.io import fits
import astropy.units as u
from utils import bin_ndarray as rebin
from utils import gauss_kern
from utils import clean_nans
from utils import clean_args
from astropy import cosmology
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import Planck15, z_at_value

class Binned_Catalog:
	#def __init__(self, tbl, zkey='z_peak', mkey='lmass', rkey='ra', dkey='dec', uvkey='rf_U_V', vjkey='rf_V_J'):
	def __init__(self, tbl, astrometry_keys = {'rkey':'ra', 'dkey':'dec'}, binning_keys = {'zkey':'z_peak', 'mkey':'lmass', 'uvkey':'rf_U_V', 'vjkey':'rf_V_J'}):

		self.table = tbl
		self.nsrc = len(tbl)
		self.astrometry_keys = astrometry_keys
		self.binning_keys = binning_keys

		pdb.set_trace()

	def separate_pops_by_name(self, cuts_dict):
		'''
		This is a generalized classifier of galaxies.
		cuts_dict is a dictionary with conditions,
		where the key is the population (e.g., 'sf', 'agn','sf0')
		E.g.;
		cuts_dict['agn'] = [pop_index,[criteria]]
		The critera contain ['key',greater-than,less-than] values with False
		when only one condition.  E.g.,
			cuts_dict['dusty'] = [3,[['lage', False, 7.5]]]
			cuts_dict['agn'] = [2,[['F_ratio', 40, False]]]
			cuts_dict['sb'] = [4,[['mips24',300,False],['lage', False, 7.5]]]

		Ncrit == len(cuts_dict) [-2 if UVJ used for sf/qt]
		Conditions should go in descending order... hard when a dictionary
		has no intrinsic order.  First operation is to determine the order and
		set conditions.
		'''
		sfg = np.ones(self.nsrc)
		npop = len(cuts_dict)
		if 'qt' in cuts_dict:
			Ncrit = npop - 2
			uvj = True
		else:
			Ncrit = npop
			uvj = False
		print (Ncrit)
		#Set (descending) order of cuts.
		#names      = [k for k in cuts_dict][::-1]
		#reverse_ind here is the arguments indices in reversed order
		reverse_ind = np.argsort([cuts_dict[k][0] for k in cuts_dict])[::-1]
		ind         = [cuts_dict[k][0] for k in cuts_dict]
		conditions  = [cuts_dict[k][1] for k in cuts_dict]
		for i in range(self.nsrc):
			# Go through conditions in descending order.
			# continue when one is satisfied
			for j in range(Ncrit):
				icut = reverse_ind[j]
				ckey = conditions[icut][0]
				if (conditions[icut][1] == False) & (conditions[icut][2] == False):
					if (self.table[ckey].values[i] == conditions[icut][3]):
						sfg[i]=ind[icut]
						continue
				elif conditions[icut][1] == False:
					if (self.table[ckey].values[i] < conditions[icut][2]):
						sfg[i]=ind[icut]
						continue
				elif conditions[icut][2] == False:
					if (self.table[ckey].values[i] > conditions[icut][1]):
						sfg[i]=ind[icut]
						continue
				else:
					if (self.table[ckey].values[i] > conditions[icut][1]) & (self.table[ckey].values[i] < conditions[icut][2]):
						sfg[i]=ind[icut]
						continue
			# If no condition yet met then see if it's Quiescent
			if (sfg[i] == 1) & (uvj == True):
					if (self.table[self.uvkey][i] > 1.3) and (self.table[self.vjkey][i] < 1.5):
						if (self.table[self.zkey][i] < 1):
							if (self.table[self.uvkey][i] > (self.table[self.vjkey][i]*0.88+0.69) ): sfg[i]=0
						if (self.table[self.zkey][i] > 1):
							if (self.table[self.uvkey][i] > (self.table[self.vjkey][i]*0.88+0.59) ): sfg[i]=0

		self.table['sfg'] = sfg

	def separate_uvj(self):
		sfg = np.ones(self.nsrc)
		#pdb.set_trace()
		for i in range(self.nsrc):
			if (self.table[self.uvkey][i] > 1.3) and (self.table[self.vjkey][i] < 1.5):
				if (self.table[self.zkey][i] < 1):
					if (self.table[self.uvkey][i] > (self.table[self.vjkey][i]*0.88+0.69) ): sfg[i]=0
				if (self.table[self.zkey][i] > 1):
					if (self.table[self.uvkey][i] > (self.table[self.vjkey][i]*0.88+0.59) ): sfg[i]=0
		self.table['sfg'] = sfg

	def separate_uvj_agn(self,Fcut = 20):
		sfg = np.ones(self.nsrc)
		#pdb.set_trace()
		#AGN
		for i in range(self.nsrc):
			if (self.table.F_ratio[i] >= Fcut):
				sfg[i]=2
			else:
				if (self.table[self.uvkey][i] > 1.3) and (self.table[self.vjkey][i] < 1.5):
					if (self.table[self.zkey][i] < 1):
						if (self.table[self.uvkey][i] > (self.table[self.vjkey][i]*0.88+0.69) ): sfg[i]=0
					if (self.table[self.zkey][i] > 1):
						if (self.table[self.uvkey][i] > (self.table[self.vjkey][i]*0.88+0.59) ): sfg[i]=0
		self.table['sfg'] = sfg


	def get_general_redshift_bins(self, znodes, mnodes, sfg = 1, suffx = '', Fcut = 25, Ahat = 1.0, initialize_pop = False):
		if initialize_pop == True: self.id_z_ms = {}
		for iz in range(len(znodes[:-1])):
			for jm in range(len(mnodes[:-1])):
				ind_mz =( (self.table.sfg == 1) & (self.table[self.zkey] >= np.min(znodes[iz:iz+2])) & (self.table[self.zkey] < np.max(znodes[iz:iz+2])) &
					(10**self.table[self.mkey] >= 10**np.min(mnodes[jm:jm+2])) & (10**self.table[self.mkey] < 10**np.max(mnodes[jm:jm+2])) )

				self.id_z_ms['z_'+clean_args(str('{:.2f}'.format(znodes[iz])))+'_'+clean_args(str('{:.2f}'.format(znodes[iz+1])))+'__m_'+clean_args(str('{:.2f}'.format(mnodes[jm])))+'_'+clean_args(str('{:.2f}'.format(mnodes[jm+1])))+suffx] = self.table.ID[ind_mz].values

	def get_mass_redshift_bins(self, znodes, mnodes, sfg = 1, pop_suffix = '', initialize_pop = False):
		if initialize_pop == True: self.id_z_ms_pop = {}
		for iz in range(len(znodes[:-1])):
			for jm in range(len(mnodes[:-1])):
				ind_mz =( (self.table.sfg == sfg) & (self.table[self.zkey] >= np.min(znodes[iz:iz+2])) & (self.table[self.zkey] < np.max(znodes[iz:iz+2])) &
					(10**self.table[self.mkey] >= 10**np.min(mnodes[jm:jm+2])) & (10**self.table[self.mkey] < 10**np.max(mnodes[jm:jm+2])) )

				self.id_z_ms_pop['z_'+clean_args(str('{:.2f}'.format(znodes[iz])))+'_'+clean_args(str('{:.2f}'.format(znodes[iz+1])))+'__m_'+clean_args(str('{:.2f}'.format(mnodes[jm])))+'_'+clean_args(str('{:.2f}'.format(mnodes[jm+1])))+pop_suffix] = self.table.ID[ind_mz].values

	def subset_positions(self,radec_ids):
		''' This positions function is very general.
			User supplies IDs dictionary, function returns RA/DEC dictionaries with the same keys'''
		ra_dec = {}
		for k in radec_ids.keys():
			ra0  = self.table[self.rkey]
			dec0 = self.table[self.dkey]
			ra  = ra0[self.table.ID.isin(radec_ids[k])].values
			dec = dec0[self.table.ID.isin(radec_ids[k])].values
			ra_dec[k] = [ra,dec]
		return ra_dec
