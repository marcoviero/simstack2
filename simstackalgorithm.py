import pdb
import os
import json
import numpy as np
from astropy.wcs import WCS
from utils import circle_mask
from utils import gauss_kern
from utils import smooth_psf
from lmfit import Parameters, minimize, fit_report
from skymaps import Skymaps
from skycatalogs import Skycatalogs
from simstacksettings import SimstackSettings

class SimstackAlgorithm(SimstackSettings, Skymaps, Skycatalogs):

    def __init__(self, param_path_file):
        super().__init__(param_path_file)

    def perform_simstack(self):

        # Get catalog.  Clean NaNs
        catalog = self.split_table.dropna()

        # Get binning details
        binning = json.loads(self.config_dict['general']['binning'])

        # Stack in redshift slices if bin_all_at_once is False
        if binning['bin_all_at_once'] == "False":
            redshifts = catalog.pop("redshift")
            for i in np.unique(redshifts):
                catalog_in = catalog[redshifts == i]
                self.stack_in_wavelengths(catalog_in, distance_interval=str(i))
        else:
            self.stack_in_wavelengths(catalog, distance_interval='all')

    def stack_in_wavelengths(self, catalog, distance_interval=None):

        map_keys = list(self.maps_dict.keys())
        for wv in map_keys:
            map_dict = self.maps_dict[wv]
            cube = self.build_cube(map_dict, catalog)
            cov_ss_1d = self.regress_cube_layers(cube)
            #pdb.set_trace()
            if 'stacked_flux_densities' not in self.maps_dict[wv]:
                self.maps_dict[wv]['stacked_flux_densities'] = {distance_interval: cov_ss_1d}
            else:
                self.maps_dict[wv]['stacked_flux_densities'][distance_interval] = cov_ss_1d

    def regress_cube_layers(self, cube):

        # Extract Noise and Signal Maps from Cube (and then delete layers)
        ierr = cube[-1, :]
        cube = cube[:-1, :]
        # Subtract mean from map
        imap = cube[-1, :] - np.mean(cube[-1, :], dtype=np.float32)
        cube = cube[:-1, :]

        fit_params = Parameters()

        print("Give these parameters real names!")
        for iarg in range(len(cube)):
            fit_params.add('layer' + str(iarg), value=1e-3 * np.random.randn())

        cov_ss_1d = minimize(self.simultaneous_stack_array_oned, fit_params,
                             args=(np.ndarray.flatten(cube),),
                             kws={'data1d': np.ndarray.flatten(imap), 'err1d': np.ndarray.flatten(ierr)})

        return cov_ss_1d

    def simultaneous_stack_array_oned(self, p, layers_1d, data1d, err1d=None, arg_order=None):
        ''' Function to Minimize written specifically for lmfit '''

        v = p.valuesdict()

        len_model = len(data1d)
        nlayers = len(layers_1d) // len_model

        model = np.zeros(len_model)

        for i in range(nlayers):
            if arg_order != None:
                model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[arg_order[i]]
            else:
                model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[list(v.keys())[i]]

        # Take the mean of the layers after they've been summed together
        model -= np.mean(model)

        if err1d is None:
            return (data1d - model)

        return (data1d - model) / err1d

    def build_cube(self, map_dict, catalog):

        cmap = map_dict['map']
        cnoise = map_dict['noise']
        pix = map_dict['pixel_size']
        hd = map_dict['header']
        fwhm = map_dict['fwhm']
        wmap = WCS(hd)

        # Extract RA and DEC from catalog
        ra_series = catalog.pop('ra')
        dec_series = catalog.pop('dec')
        keys = list(catalog.keys())

        # FIND SIZES OF MAP AND LISTS
        cms = np.shape(cmap)
        #zeromask = np.zeros(cms)

        nlists = []
        for k in keys:
            nlists.append(len(np.unique(catalog[k])))
        nlayers = np.prod(nlists)
        print("Number of Layers Stacking Simultaneously = {}".format(nlayers))

        if np.sum(cnoise) == 0: cnoise = cmap * 0.0 + 1.0

        # STEP 1  - Make Layers Cube
        layers = np.zeros([nlayers, cms[0], cms[1]])

        ilayer = 0
        ipops = np.unique(catalog[keys[0]])
        for ipop in range(nlists[0]):
            if len(nlists) > 1:
                jpops = np.unique(catalog[keys[1]])
                for jpop in range(nlists[1]):
                    if len(nlists) > 2:
                        kpops = np.unique(catalog[keys[2]])
                        for kpop in range(nlists[2]):
                            ind_src = (catalog[keys[0]] == ipops[ipop]) & (catalog[keys[1]] == jpops[jpop]) & (catalog[keys[2]] == kpops[kpop])
                            real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                            layers[ilayer, real_x, real_y] += 1.0
                            ilayer += 1
                    else:
                        ind_src = (catalog[keys[0]] == ipops[ipop]) & (catalog[keys[1]] == jpops[jpop])
                        real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                        layers[ilayer, real_x, real_y] += 1.0
                        ilayer += 1
            else:
                ind_src = (catalog[keys[0]] == ipops[ipop])
                real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                layers[ilayer, real_x, real_y] += 1.0
                ilayer += 1

        # STEP 2  - Convolve Layers and put in pixels
        radius = 1.1
        flattened_pixmap = np.sum(layers, axis=0)
        total_circles_mask = circle_mask(flattened_pixmap, radius * fwhm, pix)
        ind_fit = np.where(total_circles_mask >= 1)
        #ind_fit = np.where((total_circles_mask >= 1) & (zeromask != 0))
        nhits = np.shape(ind_fit)[1]
        cfits_maps = np.zeros([nlayers + 2, nhits])

        kern = gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
        for umap in range(nlayers):
            layer = layers[umap, :, :]
            tmap = smooth_psf(layer, kern)
            # tmap[ind_fit] -= np.mean(tmap[ind_fit])
            cfits_maps[umap, :] = tmap[ind_fit]

        # put map and noisemap in last two layers
        cfits_maps[-2, :] = cmap[ind_fit]
        cfits_maps[-1, :] = cnoise[ind_fit]

        return cfits_maps

    def get_x_y_from_ra_dec(self, wmap, cms, ind_src, ra_series, dec_series):

        ra = ra_series[ind_src].values
        dec = dec_series[ind_src].values
        # CONVERT FROM RA/DEC to X/Y
        # DANGER!!  NOTICE THAT I FLIP X AND Y HERE!!
        ty, tx = wmap.wcs_world2pix(ra, dec, 0)
        # CHECK FOR SOURCES THAT FALL OUTSIDE MAP
        ind_keep = np.where((tx >= 0) & (np.round(tx) < cms[0]) & (ty >= 0) & (np.round(ty) < cms[1]))
        nt0 = np.shape(ind_keep)[1]
        real_x = np.round(tx[ind_keep]).astype(int)
        real_y = np.round(ty[ind_keep]).astype(int)

        return real_x, real_y