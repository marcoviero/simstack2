import pdb
import os
import json
import numpy as np
from astropy.io import fits
from utils import bin_ndarray as rebin_kernel
from utils import gauss_kern
from utils import clean_nans
from utils import map_rms

class Skymaps:

	def __init__(self):
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

	def import_maps(self):

		self.maps_dict = {}
		for imap in self.config_dict['maps']:
			map_params = json.loads(self.config_dict['maps'][imap])
			if json.loads(map_params["stack"].lower()):
				map_dict = self.import_map(map_params)
				self.maps_dict[imap] = map_dict

	def import_map(self, map_dict):
		#READ MAPS

		file_map = self.parse_path(map_dict["path_map"])
		file_noise = self.parse_path(map_dict["path_noise"])
		wavelength = map_dict["wavelength"]
		psf = map_dict["beam"]["fwhm"]
		beam_area = map_dict["beam"]["area"]
		color_correction = map_dict["color_correction"]

		#SPIRE Maps have Noise maps in the second extension.
		if file_map == file_noise:
			header_ext_map = 1
			header_ext_noise = 2
		else:
			header_ext_map = 1
			header_ext_noise = 1
		if os.path.isfile(file_map) and os.path.isfile(file_noise):
			cmap, hd = fits.getdata(file_map, header_ext_map, header=True)
			cnoise, nhd = fits.getdata(file_map, header_ext_noise, header=True)
		else:
			print("Files not found, check path in config file: "+file_map)
			pdb.set_trace()

		#GET MAP PIXEL SIZE
		if 'CD2_2' in hd:
			pix = hd['CD2_2'] * 3600.
		else:
			pix = hd['CDELT2'] * 3600.

		#READ BEAMS
		#Check first if beam is a filename (actual beam) or a number (approximate with Gaussian)
		if type(psf) is str:
			beam, phd = fits.getdata(psf, 0, header=True)
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
				new_shape = (np.round(pms[0]*scale_beam), np.round(pms[1]*scale_beam))
				kern = rebin_kernel(clean_nans(beam), new_shape=new_shape, operation='ave')
			else:
				kern = clean_nans(beam)
			map_dict["psf_pixel_size"] = pix_beam
		else:
			fwhm = psf
			#sig = fwhm / 2.355 / pix
			kern = gauss_kern(psf, np.floor(fwhm * 8.)/pix, pix)

		map_dict["map"] = clean_nans(cmap) * color_correction
		map_dict["noise"] = clean_nans(cnoise, replacement_char=1e10) * color_correction
		if beam_area != 1.0:
			map_dict["map"] *= beam_area * 1e6
			map_dict["noise"] *= beam_area * 1e6
		map_dict["header"] = hd
		map_dict["pixel_size"] = pix
		map_dict["psf"] = clean_nans(kern)
		map_dict["rms"] = map_rms(map_dict["map"].copy(), silent=True)

		if wavelength != None:
			map_dict["wavelength"] = wavelength

		if fwhm != None:
			map_dict["fwhm"] = fwhm

		return map_dict
