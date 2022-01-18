import pdb
import os
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from lmfit import Parameters, minimize, fit_report
from skymaps import Skymaps
from skycatalogs import Skycatalogs
from simstacktoolbox import SimstackToolbox
from simstackresults import SimstackResults

class SimstackAlgorithm(SimstackToolbox, SimstackResults, Skymaps, Skycatalogs):

    stack_successful = False

    def __init__(self, param_path_file):
        super().__init__(param_path_file)

    def perform_simstack(self, add_background=False):

        # Get catalog.  Clean NaNs
        catalog = self.catalog_dict['tables']['split_table'].dropna()

        # Get binning details
        binning = json.loads(self.config_dict['general']['binning'])

        split_dict = json.loads(self.config_dict['catalog']['classification'])
        split_type = split_dict.pop('split_type')
        nlists = []
        for k in split_dict:
            kval = split_dict[k]['bins']
            if type(kval) is str:
                nlists.append(len(json.loads(kval))-1)  # bins so subtract 1
            elif type(kval) is dict:
                nlists.append(len(kval))
            else:
                nlists.append(kval)
        nlayers = np.prod(nlists[1:])

        # Stack in redshift slices if bin_all_at_once is False
        bins = json.loads(split_dict["redshift"]['bins'])
        distance_labels = []
        if binning['bin_all_at_once'] == "False":
            redshifts = catalog.pop("redshift")
            for i in np.unique(redshifts):
                catalog_in = catalog[redshifts == i]
                distance_label = "_".join(["redshift", str(bins[int(i)]), str(bins[int(i) + 1])]).replace('.', 'p')
                distance_labels.append(distance_label)
                labels = self.catalog_dict['tables']['parameter_labels'][int(i*nlayers):int((i+1)*nlayers)]
                if add_background:
                    labels.append("ones_background")
                self.stack_in_wavelengths(catalog_in, labels=labels, distance_interval=distance_label, add_background=add_background)
        else:
            labels = []
            for i in np.unique(catalog['redshift']):
                labels.extend(self.catalog_dict['tables']['parameter_labels'][int(i*nlayers):int((i+1)*nlayers)])
                distance_labels.append("_".join(["redshift", str(bins[int(i)]), str(bins[int(i) + 1])]).replace('.', 'p'))
            if add_background:
                labels.append("ones_background")
            self.stack_in_wavelengths(catalog, labels=labels, distance_interval='all_redshifts', add_background=add_background)

        self.config_dict['catalog']['distance_labels'] = distance_labels
        self.stack_successful = True

    def build_cube(self, map_dict, catalog, add_background=False, crop_circles=True, write_fits_layers=False):

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

        nlists = []
        for k in keys:
            nlists.append(len(np.unique(catalog[k])))
        nlayers = np.prod(nlists)

        #print("Number of Layers Stacking Simultaneously = {}".format(nlayers+np.sum(add_background)))

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
        if crop_circles:
            radius = 1.1
            flattened_pixmap = np.sum(layers, axis=0)
            total_circles_mask = self.circle_mask(flattened_pixmap, radius * fwhm, pix)
            ind_fit = np.where(total_circles_mask >= 1)
        else:
            ind_fit = np.where(0 * np.sum(layers, axis=0) == 0)

        nhits = np.shape(ind_fit)[1]
        if add_background:
            cfits_maps = np.zeros([nlayers + 3, nhits])  # +3 to append background, cmap and cnoise
        else:
            cfits_maps = np.zeros([nlayers + 2, nhits])  # +2 to append cmap and cnoise

        kern = self.gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
        for umap in range(nlayers):
            layer = layers[umap, :, :]
            tmap = self.smooth_psf(layer, kern)

            # write layers to fits files here
            if write_fits_layers:
                path_layer = r'D:\maps\cutouts\layers'
                name_layer = 'layer_'+str(umap)+'.fits'
                pdb.set_trace()
                hdu = fits.PrimaryHDU(tmap, header=hd)
                hdul = fits.HDUList([hdu])
                hdul.writeto(os.path.join(path_layer, name_layer))

            # Remove mean from map
            cfits_maps[umap, :] = tmap[ind_fit] - np.mean(tmap[ind_fit])

        # If add_background=True, add background layer of ones.
        if add_background:
            cfits_maps[-3, :] = np.ones(np.shape(cmap[ind_fit]))
        # put map and noisemap in last two layers
        cfits_maps[-2, :] = cmap[ind_fit]
        cfits_maps[-1, :] = cnoise[ind_fit]

        return cfits_maps

    def trim_label_list(self, catalog, labels):

        # Extract RA and DEC from catalog
        ra_series = catalog.pop('ra')
        dec_series = catalog.pop('dec')
        keys = list(catalog.keys())
        trimmed_labels = []

        label_dict = self.config_dict['parameter_names']
        ds = [len(label_dict[k]) for k in label_dict]
        if len(labels) == np.prod(ds[1:]):
            nlists = ds[1:]
        else:
            nlists = ds

        ilayer = 0
        for ipop in range(nlists[0]):
            if len(nlists) > 1:
                for jpop in range(nlists[1]):
                    if len(nlists) > 2:
                        for kpop in range(nlists[2]):
                            if sum((catalog[keys[0]] == ipop) & (catalog[keys[1]] == jpop) & (catalog[keys[2]] == kpop)) > 0:
                                trimmed_labels.append(labels[ilayer])
                            ilayer += 1
                    else:
                        if sum((catalog[keys[0]] == ipop) & (catalog[keys[1]] == jpop)) > 0:
                            trimmed_labels.append(labels[ilayer])
                        ilayer += 1
            else:
                if sum(catalog[keys[0]] == ipop) > 0:
                    trimmed_labels.append(labels[ilayer])
                ilayer += 1

        return trimmed_labels

    def stack_in_wavelengths(self, catalog, labels=None, distance_interval=None, crop_circles=False, add_background=False):

        map_keys = list(self.maps_dict.keys())
        for wv in map_keys:
            map_dict = self.maps_dict[wv]
            cube = self.build_cube(map_dict, catalog.copy(), crop_circles=crop_circles, add_background=add_background)
            cube_labels = self.trim_label_list(catalog.copy(), labels)
            nlayers = len(cube_labels)
            print("Simultaneously Stacking {} Layers in {}".format(nlayers, wv))
            cov_ss_1d = self.regress_cube_layers(cube, labels=cube_labels)
            if 'stacked_flux_densities' not in self.maps_dict[wv]:
                self.maps_dict[wv]['stacked_flux_densities'] = {distance_interval: cov_ss_1d}
            else:
                self.maps_dict[wv]['stacked_flux_densities'][distance_interval] = cov_ss_1d

    def regress_cube_layers(self, cube, labels=None):

        # Extract Noise and Signal Maps from Cube (and then delete layers)
        ierr = cube[-1, :]
        cube = cube[:-1, :]
        # Subtract mean from map
        imap = cube[-1, :]  # - np.mean(cube[-1, :], dtype=np.float32)
        cube = cube[:-1, :]

        # Step backward through cube so removal of rows does not affect order
        non_zero_parameter_labels = []
        ilab = 0
        for iarg in range(len(cube))[::-1]:
            # Remove empty layers
            if np.sum(abs(cube[iarg, :])) > 0:
                if not labels:
                    parameter_label = self.catalog_dict['tables']['parameter_labels'][ilab].replace('.', 'p')
                    ilab += 1
                else:
                    parameter_label = labels[ilab].replace('.', 'p')
                    ilab += 1
                # Add parameter_label to non_zero_parameter_labels
                non_zero_parameter_labels.append(parameter_label)
            else:
                # Remove empty layer from cube, and don't add to non_zero_parameter_labels
                cube = np.delete(cube, iarg, 0)

        # Add non-empty Parameters to fit_params in right order
        fit_params = Parameters()
        #for iparam in non_zero_parameter_labels[::-1]:
        for iparam in non_zero_parameter_labels:
            fit_params.add(iparam, value=1e-3 * np.random.randn())

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

        return (data1d - model)

        # Does not work anymore, whats up?
        #if (err1d is None) or 0 in err1d:
        #    return (data1d - model)
        #else:
        #    return (data1d - model)**2 / err1d**2

    def get_x_y_from_ra_dec(self, wmap, cms, ind_src, ra_series, dec_series):

        ra = ra_series[ind_src].values
        dec = dec_series[ind_src].values
        # CONVERT FROM RA/DEC to X/Y
        ty, tx = wmap.wcs_world2pix(ra, dec, 0)
        # CHECK FOR SOURCES THAT FALL OUTSIDE MAP
        ind_keep = np.where((tx >= 0) & (np.round(tx) < cms[0]) & (ty >= 0) & (np.round(ty) < cms[1]))
        real_x = np.round(tx[ind_keep]).astype(int)
        real_y = np.round(ty[ind_keep]).astype(int)

        return real_x, real_y


