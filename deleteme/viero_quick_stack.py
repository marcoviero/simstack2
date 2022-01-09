import numpy as np
import os
import readcol
from astropy.io import fits
from simstackwrapper import stack_in_redshift_slices as simstack

def viero_quick_stack(
	map_names, 
	catalog_names, 
	noise_map_names,
	efwhm=None,
	psf_names=None,
	n_sources_max = None
	): 

	if n_sources_max == None: n_sources_max=50000l
	nmap = len(map_names)

	#PUT DATA INTO CUBE
	nlists = len(catalog_names)
	nsources = 0 # initialize a counter
	cube = np.zeros([n_sources_max, nlists, 2]) # nsources by nlis/nts by 2 for RA/DEC
	for i in range(nlists): 
		list_name = catalog_names[i]
		if os.path.getsize(list_name) > 0: 
			ra, dec = readcol.readcol(list_name, fsep=',', twod=False)
			nsources_list=len(ra)
			if nsources_list > n_sources_max: 
				print 'too many sources in catalog: use N_SOURCES_MAX flag'
				break
			if nsources_list > 0:
				cube[0:nsources_list,i,0]=ra
				cube[0:nsources_list,i,1]=dec
			if nsources_list > nsources: 
				nsources=nsources_list

	cube=cube[0:nsources,:,:] # Crop it down to the length of longest list

	stacked_sed=np.zeros([nmap, nlists])
	stacked_sed_err=np.zeros([nmap,nlists])

	#USE SIMSTACK TO STACK AT ONE WAVELENGTH AT A TIME
	for wv in range(nmap):
		#READ MAPS
		cmap, chd = fits.getdata(map_names[wv], 0, header = True)

		if noise_map_names != None:
			cnoise, nhd = fits.getdata(noise_map_names[wv], 0, header = True)

		if efwhm != None:
			ifwhm = efwhm[wv]
			ipsf_names = None
		else:
			ifwhm = None
			ipsf_names = psf_names[wv]

		stacked_object=simstack(
			cmap,
			chd,
			cube,
			fwhm=ifwhm,
			psf_names=ipsf_names,
			cnoise=cnoise,
		#	err_ss=err_ss,
			quiet=None)

		stacked_flux = np.array(stacked_object.params.values())
		stacked_sed[wv,:] = stacked_flux
		#stacked_sed_err[wv,:]=err_ss
	#pdb.set_trace()
	return stacked_sed