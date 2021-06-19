import pdb
import os
import six
import numpy as np
from astropy.io import fits
#import astropy.units as u
from utils import bin_ndarray as rebin
from utils import gauss_kern
from utils import clean_nans
#from utils import clean_args
from utils import map_rms
#from astropy import cosmology
#from astropy.cosmology import Planck15 as cosmo
#from astropy.cosmology import Planck15, z_at_value

class Skymaps:

	def __init__(self,file_map,file_noise,psf,color_correction=1.0,beam_area=1.0,wavelength=None,fwhm=None):
		''' This Class creates Objects for a set of
		maps/noisemaps/beams/TransferFunctions/etc.,
		at each Wavelength.
		This is a work in progress!
		Issues:  If the beam has a different pixel size from the map,
		it is not yet able to re-scale it.
		Just haven't found a convincing way to make it work.
		Future Work:
		Will shift some of the work into functions (e.g., read psf,
		color_correction) and increase flexibility.
		'''
		#READ MAPS

		if file_map == file_noise:
			#SPIRE Maps have Noise maps in the second extension.
			cmap, hd = fits.getdata(os.path.join(os.environ['MAPSPATH'],file_map), 1, header = True)
			cnoise, nhd = fits.getdata(os.path.join(os.environ['MAPSPATH'],file_map), 2, header = True)
		else:
			#This assumes that if Signal and Noise are different maps, they are contained in first extension
			cmap, hd = fits.getdata(os.path.join(os.environ['MAPSPATH'],file_map), 0, header = True)
			cnoise, nhd = fits.getdata(os.path.join(os.environ['MAPSPATH'],file_noise), 0, header = True)

		#GET MAP PIXEL SIZE
		if 'CD2_2' in hd:
			pix = hd['CD2_2'] * 3600.
		else:
			pix = hd['CDELT2'] * 3600.

		#READ BEAMS
		#Check first if beam is a filename (actual beam) or a number (approximate with Gaussian)
		if isinstance(psf, six.string_types):
			beam, phd = fits.getdata(psf, 0, header = True)
			#GET PSF PIXEL SIZE
			if 'CD2_2' in phd:
				pix_beam = phd['CD2_2'] * 3600.
			elif 'CDELT2' in phd:
				pix_beam = phd['CDELT2'] * 3600.
			else: pix_beam = pix
			#SCALE PSF IF NECESSARY
			if np.round(10.*pix_beam) != np.round(10.*pix):
				raise ValueError("Beam and Map have different size pixels")
				scale_beam = pix_beam / pix
				pms = np.shape(beam)
				new_shape=(np.round(pms[0]*scale_beam),np.round(pms[1]*scale_beam))
				#pdb.set_trace()
				kern = rebin(clean_nans(beam),new_shape=new_shape,operation='ave')
				#kern = rebin(clean_nans(beam),new_shape[0],new_shape[1])
			else:
				kern = clean_nans(beam)
			self.psf_pixel_size = pix_beam
		else:
			sig = psf / 2.355 / pix
			#pdb.set_trace()
			#kern = gauss_kern(psf, np.floor(psf * 8.), pix)
			kern = gauss_kern(psf, np.floor(psf * 8.)/pix, pix)

		self.map = clean_nans(cmap) * color_correction
		self.noise = clean_nans(cnoise,replacement_char=1e10) * color_correction
		if beam_area != 1.0:
			self.beam_area_correction(beam_area)
		self.header = hd
		self.pixel_size = pix
		self.psf = clean_nans(kern)
		self.rms = map_rms(self.map.copy(), silent=True)

		if wavelength != None:
			add_wavelength(wavelength)

		if fwhm != None:
			add_fwhm(fwhm)

	def beam_area_correction(self,beam_area):
		self.map *= beam_area * 1e6
		self.noise *= beam_area * 1e6

	def add_wavelength(self,wavelength):
		self.wavelength = wavelength

	def add_fwhm(self,fwhm):
		self.fwhm = fwhm

	def add_weights(self,file_weights):
		weights, whd = fits.getdata(file_weights, 0, header = True)
		#pdb.set_trace()
		self.noise = clean_nans(1./weights,replacement_char=1e10)
