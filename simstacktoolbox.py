import pdb
import os
import shutil
import logging
import pickle
import numpy as np
from configparser import ConfigParser
from lmfit import Parameters, minimize, fit_report
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

class SimstackToolbox:

    config_dict = {}

    def __init__(self, param_file_path):

        self.config_dict = self.get_params_dict(param_file_path)

    def fast_sed_fitter(self, wavelengths, fluxes, covar, betain=1.8):
        fit_params = Parameters()
        fit_params.add('A', value=1e-32, vary=True)
        fit_params.add('T_observed', value=24.0, vary=True, min=0.1)
        fit_params.add('beta', value=betain, vary=False)
        fit_params.add('alpha', value=2.0, vary=False)

        # nu_in = c * 1.e6 / wavelengths

        sed_params = minimize(self.find_sed_min, fit_params,
                              args=(np.ndarray.flatten(wavelengths),),
                              kws={'fluxes': fluxes, 'covar': covar})

        m = sed_params.params
        # m = sed_params

        return m

    def find_sed_min(self, p, wavelengths, fluxes, covar=None):

        graybody = self.fast_sed(p, wavelengths)
        # print p['T_observed']
        # print fluxes - graybody
        if covar == None:
            return (fluxes - graybody)
        else:
            return (fluxes - graybody) / covar


    def fast_sed(self, m, wavelengths):
        nu_in = c * 1.e6 / wavelengths

        v = m.valuesdict()
        A = np.asarray(v['A'])
        T = np.asarray(v['T_observed'])
        betain = np.asarray(v['beta'])
        alphain = np.asarray(v['alpha'])
        ng = np.size(A)

        ns = len(nu_in)
        base = 2.0 * (6.626) ** (-2.0 - betain - alphain) * (1.38) ** (3. + betain + alphain) / (2.99792458) ** 2.0
        expo = 34.0 * (2.0 + betain + alphain) - 23.0 * (3.0 + betain + alphain) - 16.0 + 26.0
        K = base * 10.0 ** expo
        w_num = A * K * (T * (3.0 + betain + alphain)) ** (3.0 + betain + alphain)
        w_den = (np.exp(3.0 + betain + alphain) - 1.0)
        w_div = w_num / w_den
        nu_cut = (3.0 + betain + alphain) * 0.208367e11 * T

        graybody = np.reshape(A, (ng, 1)) * nu_in ** np.reshape(betain, (ng, 1)) * black(nu_in, T) / 1000.0
        powerlaw = np.reshape(w_div, (ng, 1)) * nu_in ** np.reshape(-1.0 * alphain, (ng, 1))
        graybody[np.where(nu_in >= np.reshape(nu_cut, (ng, 1)))] = powerlaw[
            np.where(nu_in >= np.reshape(nu_cut, (ng, 1)))]

        return graybody

    def clean_nans(self, dirty_array, replacement_char=0.0):
        clean_array = dirty_array.copy()
        clean_array[np.isnan(dirty_array)] = replacement_char
        clean_array[np.isinf(dirty_array)] = replacement_char

        return clean_array

    def gauss(self, x, x0, y0, sigma):
        p = [x0, y0, sigma]
        return p[1] * np.exp(-((x - p[0]) / p[2]) ** 2)

    def gauss_kern(self, fwhm, side, pixsize):
        ''' Create a 2D Gaussian (size= side x side)'''

        sig = fwhm / 2.355 / pixsize
        delt = np.zeros([int(side), int(side)])
        delt[0, 0] = 1.0
        ms = np.shape(delt)
        delt = self.shift_twod(delt, ms[0] / 2, ms[1] / 2)
        kern = delt
        gaussian_filter(delt, sig, output=kern)
        kern /= np.max(kern)

        return kern

    def shift_twod(self, seq, x, y):
        out = np.roll(np.roll(seq, int(x), axis=1), int(y), axis=0)
        return out

    def smooth_psf(self, mapin, psfin):

        s = np.shape(mapin)
        mnx = s[0]
        mny = s[1]

        s = np.shape(psfin)
        pnx = s[0]
        pny = s[1]

        psf_x0 = pnx / 2
        psf_y0 = pny / 2
        psf = psfin
        px0 = psf_x0
        py0 = psf_y0

        # pad psf
        psfpad = np.zeros([mnx, mny])
        psfpad[0:pnx, 0:pny] = psf

        # shift psf so that centre is at (0,0)
        psfpad = self.shift_twod(psfpad, -px0, -py0)
        smmap = np.real(np.fft.ifft2(np.fft.fft2(mapin) *
                                     np.fft.fft2(psfpad))
                        )

        return smmap

    def dist_idl(self, n1, m1=None):
        ''' Copy of IDL's dist.pro
        Create a rectangular array in which each element is
        proportinal to its frequency'''

        if m1 == None:
            m1 = int(n1)

        x = np.arange(float(n1))
        for i in range(len(x)): x[i] = min(x[i], (n1 - x[i])) ** 2.

        a = np.zeros([int(n1), int(m1)])

        i2 = m1 // 2 + 1

        for i in range(i2):
            y = np.sqrt(x + i ** 2.)
            a[:, i] = y
            if i != 0:
                a[:, m1 - i] = y

        return a

    def circle_mask(self, pixmap, radius_in, pixres):
        ''' Makes a 2D circular image of zeros and ones'''

        radius = radius_in / pixres
        xy = np.shape(pixmap)
        xx = xy[0]
        yy = xy[1]
        beforex = np.log2(xx)
        beforey = np.log2(yy)
        if beforex != beforey:
            if beforex > beforey:
                before = beforex
            else:
                before = beforey
        else:
            before = beforey
        l2 = np.ceil(before)
        pad_side = int(2.0 ** l2)
        outmap = np.zeros([pad_side, pad_side])
        outmap[:xx, :yy] = pixmap

        dist_array = np.shift_twod(np.dist_idl(pad_side, pad_side), pad_side / 2, pad_side / 2)
        circ = np.zeros([pad_side, pad_side])
        ind_one = np.where(dist_array <= radius)
        circ[ind_one] = 1.
        mask = np.real(np.fft.ifft2(np.fft.fft2(circ) *
                                    np.fft.fft2(outmap))
                       ) * pad_side * pad_side
        mask = np.round(mask)
        ind_holes = np.where(mask >= 1.0)
        mask = mask * 0.
        mask[ind_holes] = 1.
        maskout = shift_twod(mask, pad_side / 2, pad_side / 2)

        return maskout[:xx, :yy]

    def map_rms(self, map, mask=None):
        if mask != None:
            ind = np.where((mask == 1) & (self.clean_nans(map) != 0))
            print('using mask')
        else:
            ind = self.clean_nans(map) != 0
        map /= np.max(map)

        x0 = abs(np.percentile(map, 99))
        hist, bin_edges = np.histogram(np.unique(map), range=(-x0, x0), bins=30, density=True)

        p0 = [0., 1., x0 / 3]
        x = .5 * (bin_edges[:-1] + bin_edges[1:])
        x_peak = 1 + np.where((hist - max(hist)) ** 2 < 0.01)[0][0]

        # Fit the data with the function
        fit, tmp = curve_fit(self.gauss, x[:x_peak], hist[:x_peak] / max(hist), p0=p0)
        rms_1sig = abs(fit[2])

        return rms_1sig

    def parse_path(self, path_in):
        #print(path_in)
        path_in = path_in.split(" ")
        if len(path_in) == 1:
            return path_in[0]
        else:
            path_env = os.environ[path_in[0]]
            if len(path_in) == 2:
                if 'nt' in os.name:
                    return path_env + os.path.join('\\', path_in[1].replace('/', '\\'))
                else:
                    return path_env + os.path.join('/', path_in[1])
            else:
                if 'nt' in os.name:
                    path_rename = [i.replace('/', '\\') for i in path_in[1:]]
                    return path_env + os.path.join('\\', *path_rename)
                else:
                    return path_env + os.path.join('/', *path_in[1:])

    def import_saved_pickles(self, pickle_fn):
        with open(pickle_fn, "rb") as file_path:
            encoding = pickle.load(file_path)
        return encoding

    def save_stacked_fluxes(self, fp_in, append_to_existing=False):
        if 'shortname' in self.config_dict['io']:
            shortname = self.config_dict['io']['shortname']
        else:
            shortname = os.path.basename(fp_in).split('.')[0]

        out_file_path = os.path.join(self.parse_path(self.config_dict['io']['output_folder']),
                                     shortname)
        if not os.path.exists(out_file_path):
            os.makedirs(out_file_path)
        else:
            if not append_to_existing:
                while os.path.exists(out_file_path):
                    out_file_path = out_file_path + "_"
                os.makedirs(out_file_path)

        fpath = os.path.join(out_file_path, shortname + '.pkl')

        print('pickling to ' + fpath)
        self.fpath=fpath
        with open(fpath, "wb") as pickle_file_path:
            pickle.dump(self, pickle_file_path)

        # Copy Parameter File
        fp_name = os.path.basename(fp_in)
        fp_out = os.path.join(out_file_path, fp_name)
        logging.info("Copying parameter file...")
        logging.info("  FROM : {}".format(fp_in))
        logging.info("    TO : {}".format(fp_out))
        logging.info("")
        shutil.copyfile(fp_in, fp_out)

    def get_params_dict(self, param_file_path):
        config = ConfigParser()
        config.read(param_file_path)

        dict_out = {}
        for section in config.sections():
            dict_sect = {}
            for (each_key, each_val) in config.items(section):
                dict_sect[each_key] = each_val.replace("'", '"')

            dict_out[section] = dict_sect

        return dict_out

    def write_config_file(self, params_out, config_filename_out):
        config_out = ConfigParser()

        for ikey, idict in params_out.items():
            if not config_out.has_section(ikey):

                config_out.add_section(ikey)
                for isect, ivals in idict.items():
                    # pdb.set_trace()
                    # print(ikey, isect, ivals)
                    config_out.set(ikey, isect, str(ivals))

        # Write config_filename_out (check if overwriting externally)
        with open(config_filename_out, 'w') as conf:
            config_out.write(conf)