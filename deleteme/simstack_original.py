import pdb
import numpy as np
import gc
#import cPickle as pickle
from astropy.wcs import WCS
from utils import circle_mask
from utils import clean_args
from utils import clean_nans
from utils import gauss_kern
from utils import smooth_psf
from lmfit import Parameters, minimize

pi = 3.141592653589793
L_sun = 3.839e26  # W
c = 299792458.0  # m/s
conv_sfr = 1.728e-10 / 10 ** (.23)
conv_luv_to_sfr = 2.17e-10
conv_lir_to_sfr = 1.72e-10
a_nu_flux_to_mass = 6.7e19
flux_to_specific_luminosity = 1.78  # 1e-23 #1.78e-13
h = 6.62607004e-34  # m2 kg / s  #4.13e-15 #eV/s
k = 1.38064852e-23  # m2 kg s-2 K-1 8.617e-5 #eV/K


def simultaneous_stack_array_oned(p, layers_1d, data1d, err1d=None, arg_order=None):
    ''' Function to Minimize written specifically for lmfit '''

    v = p.valuesdict()

    len_model = len(data1d)
    nlayers = len(layers_1d) // len_model

    model = np.zeros(len_model)

    for i in range(nlayers):
        # print(v.keys()[i])
        # print(arg_order[i])
        # pdb.set_trace()
        if arg_order != None:
            model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[arg_order[i]]
        else:
            model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[list(v.keys())[i]]

    # Take the mean of the layers after they've been summed together
    model -= np.mean(model)

    if err1d is None:
        return (data1d - model)
    return (data1d - model) / err1d


def stack_in_redshift_slices(
        cmap,
        hd,
        layers_radec,
        fwhm=None,
        psf_names=None,
        cnoise=None,
        mask=None,
        beam_area=None,
        err_ss=None,
        quiet=None):
    ''' The first iteration of the translation from IDL to Python.
      Looks like an IDL function.
      Suggest using wrappers like viero_quick_stack.py
      but highly recommend Pythonic: stack_dictionaries_in_layers
      function that can be found below.
  '''

    w = WCS(hd)
    # FIND SIZES OF MAP AND LISTS
    cms = np.shape(cmap)
    zeromask = np.zeros(cms)

    size_cube = np.shape(layers_radec)
    nsrcmax = size_cube[0]
    nlists = int(size_cube[1])

    ind_map_zero = np.where(np.isnan(cmap))
    nzero = np.shape(ind_map_zero)[1]

    if np.sum(cnoise) == 0: cnoise = cmap * 0.0 + 1.0

    pix = hd["CD2_2"] * 3600.
    if pix == 0: pix = hd["CDELT2"] * 3600.

    # [STEP 0] - Calibrate maps
    if beam_area != None:
        cmap = cmap * beam_area * 1e6
        cnoise = noise * beam_area * 1e6

    # STEP 1  - Make Layers Cube
    layers = np.zeros([nlists, cms[0], cms[1]])

    for s in range(nlists):
        ind_src = np.where(layers_radec[:, s, 0] != 0)
        if np.shape(ind_src)[1] > 0:
            ra = layers_radec[ind_src, s, 0]
            dec = layers_radec[ind_src, s, 1]
            # CONVERT FROM RA/DEC to X/Y
            # DANGER!!  NOTICE THAT I FLIP X AND Y HERE!!
            ty, tx = w.wcs_world2pix(ra, dec, 0)
            # CHECK FOR SOURCES THAT FALL OUTSIDE MAP
            ind_keep = np.where((tx[0] >= 0) & (np.round(tx[0]) < cms[0]) & (ty[0] >= 0) & (np.round(ty[0]) < cms[1]))
            nt0 = np.shape(ind_keep)[1]
            real_x = np.round(tx[0, ind_keep][0]).astype(int)
            real_y = np.round(ty[0, ind_keep][0]).astype(int)
            # CHECK FOR SOURCES THAT FALL ON ZEROS MAP
            if nzero > 0:
                tally = np.zeros(nt0)
                for d in range(nt0):
                    if cmap[real_x[d], real_y[d]] != 0:
                        tally[d] = 1.
                ind_nz = np.where(tally == 1)
                nt = np.shape(ind_nz)[1]
                real_x = real_x[ind_nz]
                real_y = real_y[ind_nz]
            else:
                nt = nt0
            for ni in range(nt):
                layers[s, real_x[ni], real_y[ni]] += 1.0

    # STEP 2  - Convolve Layers and put in pixels
    radius = 1.1
    sig = fwhm / 2.355 / pix
    flattened_pixmap = np.sum(layers, axis=0)
    total_circles_mask = circle_mask(flattened_pixmap, radius * fwhm, pix)
    # ind_fit = np.where(total_circles_mask >= 1)
    ind_fit = np.where((total_circles_mask >= 1) & (zeromask != 0))
    nhits = np.shape(ind_fit)[1]
    cfits_maps = np.zeros([nlists, nhits])

    # kern = gauss_kern(fwhm, np.floor(fwhm * 10), pix)
    pdb.set_trace()
    kern = gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
    for u in range(nlists):
        layer = layers[u, :, :]
        tmap = smooth_psf(layer, kern)
        # tmap[ind_fit] -= np.mean(tmap[ind_fit])
        cfits_maps[u, :] = tmap[ind_fit]

    # STEP 3 - Regress Layers with Map (i.e., stack!)

    cmap[ind_fit] -= np.mean(cmap[ind_fit], dtype=np.float32)

    fit_params = Parameters()

    for iarg in range(nlists):
        fit_params.add('layer' + str(iarg), value=1e-3 * np.random.randn())
    imap = cmap[ind_fit]
    ierr = cnoise[ind_fit]

    cov_ss_1d = minimize(simultaneous_stack_array_oned, fit_params,
                         args=(np.ndarray.flatten(cfits_maps),),
                         kws={'data1d': np.ndarray.flatten(imap), 'err1d': np.ndarray.flatten(ierr)})

    return cov_ss_1d


def stack_dictionaries_in_layers(
        map_dictionary,
        subcatalog_dictionary,
        quiet=None):
    map_names = [i for i in map_dictionary.keys()]
    # All wavelengths in cwavelengths
    cwavelengths = [map_dictionary[i].wavelength for i in map_names]
    # Unique wavelengths in uwavelengths
    uwavelengths = np.sort(np.unique(cwavelengths))
    # nwv the number of unique wavelengths
    nwv = len(uwavelengths)

    lists = subcatalog_dictionary.keys()
    nlists = len(lists)
    # stacked_sed=np.zeros([nwv, nlists])
    # stacked_sed_err=np.zeros([nwv,nlists])
    stacked_layers = {}
    signal_to_noise = {}

    cwavelengths = []
    radius = 1.1
    for iwv in range(nwv):
        #pdb.set_trace()
        ikey = list(map_dictionary.keys())[iwv]
        print('stacking ' + ikey)
        # READ MAPS
        cmap = map_dictionary[ikey].map
        cnoise = map_dictionary[ikey].noise
        cwv = map_dictionary[ikey].wavelength
        crms = map_dictionary[ikey].rms
        cname = ikey
        cwavelengths.append(cwv)
        chd = map_dictionary[ikey].header
        pixsize = map_dictionary[ikey].pixel_size
        kern = map_dictionary[ikey].psf
        fwhm = map_dictionary[ikey].fwhm
        cw = WCS(chd)
        cms = np.shape(cmap)
        zeromask = np.ones(np.shape(cmap))
        # ind_map_zero = np.where(np.isnan(cmap))
        ind_map_zero = np.where(clean_nans(cmap) == 0.0)
        zeromask[ind_map_zero] = 0.0

        # STEP 1  - Make Layers Cube at each wavelength
        layers = np.zeros([nlists, cms[0], cms[1]])
        ngals_layer = {}

        for k in range(nlists):
            s = list(lists)[k]
            #pdb.set_trace()

            if len(subcatalog_dictionary[s][0]) > 0:
                ra = subcatalog_dictionary[s][0]
                dec = subcatalog_dictionary[s][1]
                ty, tx = cw.wcs_world2pix(ra, dec, 0)
                # CHECK FOR SOURCES THAT FALL OUTSIDE MAP
                ind_keep = np.where(
                    (np.round(tx) >= 0) & (np.round(tx) < cms[0]) & (np.round(ty) >= 0) & (np.round(ty) < cms[1]))
                real_x = np.round(tx[ind_keep]).astype(int)
                real_y = np.round(ty[ind_keep]).astype(int)
                # CHECK FOR SOURCES THAT FALL ON ZEROS
                ind_nz = np.where(cmap[real_x, real_y] != 0)
                nt = np.shape(ind_nz)[1]
                ngals_layer[s] = nt
                # print('ngals: ' + str(nt))
                if nt > 0:
                    real_x = real_x[ind_nz]
                    real_y = real_y[ind_nz]

                    for ni in range(nt):
                        layers[k, real_x[ni], real_y[ni]] += 1.0
            else:
                ngals_layer[s] = 1

        # STEP 2  - Convolve Layers and put in pixels
        flattened_pixmap = np.sum(layers, axis=0)
        total_circles_mask = circle_mask(flattened_pixmap, radius * fwhm, pixsize)
        # ind_fit = np.where(total_circles_mask >= 1)
        ind_fit = np.where((total_circles_mask >= 1) & (zeromask != 0))
        del total_circles_mask
        nhits = np.shape(ind_fit)[1]
        ###
        cfits_flat = np.asarray([])
        ###

        # print(cms)
        # pdb.set_trace()
        for u in range(nlists):
            layer = layers[u, :, :]
            # tmap = pad_and_smooth_psf(layer, kern)
            # pdb.set_trace()
            tmap = smooth_psf(layer, kern)
            # Commented out next line, which is removal of mean, and replaced w/
            # summed mean removal in simultaneous_stack_array_oned.
            # Reason is because want faint sources to potentially be negative in
            # mean-subtraced map
            # tmap[ind_fit] -= np.mean(tmap[ind_fit])
            cfits_flat = np.append(cfits_flat, np.ndarray.flatten(tmap[ind_fit]))

        cmap[ind_fit] -= np.mean(cmap[ind_fit], dtype=np.float32)
        imap = np.ndarray.flatten(cmap[ind_fit])
        ierr = np.ndarray.flatten(cnoise[ind_fit])

        fit_params = Parameters()
        # pdb.set_trace()
        for iarg in range(nlists):
            arg = clean_args(list(lists)[iarg])
            fit_params.add(arg, value=1e-3 * np.random.randn())

        if len(ierr) == 0: pdb.set_trace()
        cov_ss_1d = minimize(simultaneous_stack_array_oned, fit_params,
                             args=(cfits_flat,), kws={'data1d': imap, 'err1d': ierr}, nan_policy='propagate')
        del cfits_flat, imap, ierr

        # Dictionary keys decided here.  Was originally wavelengths.  Changing it back to map_names
        # packed_fluxes = pack_fluxes(cov_ss_1d.params)
        packed_stn = pack_simple_poisson_errors(cov_ss_1d.params, ngals_layer, crms)
        # pdb.set_trace()
        stacked_layers[cname] = packed_stn  # packed_fluxes

    gc.collect
    return stacked_layers


def stack_dictionaries_in_layers_w_background(
        map_dictionary,
        subcatalog_dictionary,
        quiet=None):
    print('stacking with floating background')
    map_names = [i for i in map_dictionary.keys()]
    # All wavelengths in cwavelengths
    cwavelengths = [map_dictionary[i].wavelength for i in map_names]
    # Unique wavelengths in uwavelengths
    uwavelengths = np.sort(np.unique(cwavelengths))
    # nwv the number of unique wavelengths
    nwv = len(uwavelengths)

    lists = subcatalog_dictionary.keys()
    nlists = len(lists)
    stacked_layers = {}

    cwavelengths = []
    radius = 1.1
    for iwv in range(nwv):
        print('stacking ' + map_dictionary.keys()[iwv])
        # READ MAPS
        cmap = map_dictionary[map_dictionary.keys()[iwv]].map
        cnoise = map_dictionary[map_dictionary.keys()[iwv]].noise
        cwv = map_dictionary[map_dictionary.keys()[iwv]].wavelength
        cname = map_dictionary.keys()[iwv]
        cwavelengths.append(cwv)
        chd = map_dictionary[map_dictionary.keys()[iwv]].header
        pixsize = map_dictionary[map_dictionary.keys()[iwv]].pixel_size
        kern = map_dictionary[map_dictionary.keys()[iwv]].psf
        fwhm = map_dictionary[map_dictionary.keys()[iwv]].fwhm
        cw = WCS(chd)
        cms = np.shape(cmap)
        zeromask = np.ones(np.shape(cmap))
        # ind_map_zero = np.where(np.isnan(cmap))
        ind_map_zero = np.where(clean_nans(cmap) == 0.0)
        zeromask[ind_map_zero] = 0.0
        # pdb.set_trace()

        # STEP 1  - Make Layers Cube at each wavelength
        layers = np.zeros([nlists + 1, cms[0], cms[1]])
        ngals_layer = {}

        for k in range(nlists):
            s = lists[k]
            if len(subcatalog_dictionary[s][0]) > 0:
                ra = subcatalog_dictionary[s][0]
                dec = subcatalog_dictionary[s][1]
                ty, tx = cw.wcs_world2pix(ra, dec, 0)
                # CHECK FOR SOURCES THAT FALL OUTSIDE MAP
                ind_keep = np.where(
                    (np.round(tx) >= 0) & (np.round(tx) < cms[0]) & (np.round(ty) >= 0) & (np.round(ty) < cms[1]))
                real_x = np.round(tx[ind_keep]).astype(int)
                real_y = np.round(ty[ind_keep]).astype(int)
                # CHECK FOR SOURCES THAT FALL ON ZEROS
                ind_nz = np.where(cmap[real_x, real_y] != 0)
                nt = np.shape(ind_nz)[1]
                ngals_layer[s] = nt
                # print('ngals: ' + str(nt))
                if nt > 0:
                    real_x = real_x[ind_nz]
                    real_y = real_y[ind_nz]

                    for ni in range(nt):
                        layers[k, real_x[ni], real_y[ni]] += 1.0
            else:
                ngals_layer[s] = 1

        # STEP 2  - Convolve Layers and put in pixels
        flattened_pixmap = np.sum(layers, axis=0)
        total_circles_mask = circle_mask(flattened_pixmap, radius * fwhm, pixsize)
        # ind_fit = np.where(total_circles_mask >= 1)
        ind_fit = np.where((total_circles_mask >= 1) & (zeromask != 0))
        del total_circles_mask
        nhits = np.shape(ind_fit)[1]
        ###
        cfits_flat = np.asarray([])
        cfits_flat = np.append(cfits_flat, np.ndarray.flatten(np.ones(len(ind_fit))))
        ###
        # pdb.set_trace()
        for u in range(nlists):
            layer = layers[u, :, :]
            tmap = smooth_psf(layer, kern)
            cfits_flat = np.append(cfits_flat, np.ndarray.flatten(tmap[ind_fit]))

        cmap[ind_fit] -= np.mean(cmap[ind_fit], dtype=np.float32)
        imap = np.ndarray.flatten(cmap[ind_fit])
        ierr = np.ndarray.flatten(cnoise[ind_fit])

        fit_params = Parameters()
        fit_params.add('cib_background', value=1e-5 * np.random.randn())
        for iarg in range(nlists):
            arg = clean_args(lists[iarg])
            fit_params.add(arg, value=1e-3 * np.random.randn())

        if len(ierr) == 0: pdb.set_trace()
        cov_ss_1d = minimize(simultaneous_stack_array_oned, fit_params,
                             args=(cfits_flat,), kws={'data1d': imap, 'err1d': ierr}, nan_policy='propagate')
        del cfits_flat, imap, ierr

        # Dictionary keys decided here.  Was originally wavelengths.  Changing it back to map_names
        packed_fluxes = pack_fluxes(cov_ss_1d.params)
        packed_stn = pack_simple_poisson_errors(cov_ss_1d.params, ngals_layer, crms)
        stacked_layers[cname] = packed_stn  # packed_fluxes

    gc.collect
    return stacked_layers


def pack_fluxes(input_params):
    packed_fluxes = {}
    for iparam in input_params:
        packed_fluxes[iparam] = {}
        packed_fluxes[iparam]['value'] = input_params[iparam].value
        packed_fluxes[iparam]['stderr'] = input_params[iparam].stderr

    return packed_fluxes


def pack_simple_poisson_errors(input_params, ngals, map_rms):
    packed_stn = {}
    for iparam in input_params:
        packed_stn[iparam] = {}
        packed_stn[iparam]['value'] = input_params[iparam].value
        packed_stn[iparam]['stderr'] = input_params[iparam].stderr
        packed_stn[iparam]['ngals_bin'] = ngals[iparam]
        packed_stn[iparam]['psnerr'] = map_rms / np.sqrt(float(ngals[iparam]))

    # pdb.set_trace()
    return packed_stn


def is_true(raw_params, key):
    """Is raw_params[key] true? Returns boolean value.
    """
    sraw = raw_params[key]
    s = sraw.lower()  # Make case-insensitive

    # Lists of acceptable 'True' and 'False' strings
    true_strings = ['true', 't', 'yes', 'y', '0']
    false_strings = ['false', 'f', 'no', 'n', '-1']
    if s in true_strings:
        return True
    elif s in false_strings:
        return False
    else:
        logging.warning("Input not recognized for parameter: %s" % (key))
        logging.warning("You provided: %s" % (sraw))
        raise
