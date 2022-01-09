import numpy as np
import os
import os.path
#import cPickle as pickle
import pickle
from astropy.cosmology import Planck15 as cosmo
from deleteme import parameters
from deleteme.utils import clean_args

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


class PickledStacksReader:
    '''A class to read and organize the output of simstack.  Point it to the location of
	the output directory and name of parameter file, and it will determine if it's
	reading stacks or bootstraps, and organizes the outputs into N-dimensional arrays.
	'''

    def __init__(self, config_path, config_file, ndecimal=2, cosmo=cosmo, area_deg=1.62):
        ''' Uses the config_file to determine if reading in bootstraps or not.
		'''

        self.path = config_path
        self.config_file = config_file
        self.params = self.get_parameters(config_path + config_file)
        if self.params['bootstrap'] == True:
            self.nboots = int(self.params['number_of_boots'])
        try:
            try:
                indpop = np.argsort(np.array([i for i in self.params['populations']['pop_names']]))
            except:
                indpop = np.argsort(np.array([i for i in self.params['populations'].values()]))
        except:
            indpop = np.argsort(np.array([i[0] for i in self.params['populations'].values()]))
        try:
            self.pops = [self.params['populations']['pop_names'][i] for i in indpop]
        except:
            self.pops = [self.params['populations'].keys()[i] for i in indpop]
        self.npops = len(self.pops)
        self.nz = len(self.params['bins']['z_nodes']) - 1
        self.nm = len(self.params['bins']['m_nodes']) - 1
        self.nw = len(self.params['map_files'])
        self.ind = np.argsort(np.array([self.params['wavelength'][wv] for wv in self.params['wavelength']]))
        self.maps = [self.params['wavelength'].keys()[i] for i in self.ind]
        self.wvs = [self.params['wavelength'].values()[i] for i in self.ind]
        self.fqs = [c * 1.e6 / self.params['wavelength'].values()[i] for i in self.ind]
        self.z_nodes = self.params['bins']['z_nodes']
        self.m_nodes = self.params['bins']['m_nodes']
        if self.params['bins']['bin_in_lookback_time'] == True:
            self.ndec = ndecimal
        else:
            self.ndec = ndecimal
        z_m_keys = self.m_z_key_builder(ndecimal=self.ndec)
        self.z_keys = z_m_keys[0]
        self.m_keys = z_m_keys[1]
        self.bin_ids = {}
        self.read_pickles()
        if self.params['bootstrap'] == True:
            ax = len(np.shape(self.bootstrap_flux_array)) - 1
            self.boot_error_bars = np.sqrt(np.var(self.bootstrap_flux_array, axis=ax))

        # self.covariance =

    def get_error_bar_dictionary(self):
        print('return a dictionary with nodes for keys')

    def get_parameters(self, config_path):

        params = parameters.get_params(config_path)

        return params

    def read_pickles(self):

        if self.params['bootstrap'] == True:
            print('creating bootstrap array w/ size ' + str(self.nw) + 'bands; ' + str(self.nz) + 'redshifts; ' + str(
                self.nm) + 'masses; ' + str(self.npops) + 'populations; ' + str(self.nboots) + ' bootstraps')
            bootstrap_fluxes = np.zeros([self.nw, self.nz, self.nm, self.npops, self.nboots])
            bootstrap_errors = np.zeros([self.nw, self.nz, self.nm, self.npops, self.nboots])
            bootstrap_intensities = np.zeros([self.nw, self.nz, self.nm, self.npops, self.nboots])
        else:
            print('creating simstack array w/ size ' + str(self.nw) + 'bands; ' + str(self.nz) + 'redshifts; ' + str(
                self.nm) + 'masses; ' + str(self.npops) + 'populations')
            stacked_fluxes = np.zeros([self.nw, self.nz, self.nm, self.npops])
            stacked_errors = np.zeros([self.nw, self.nz, self.nm, self.npops])
            stacked_intensities = np.zeros([self.nw, self.nz, self.nm, self.npops])
        if self.params['bins']['bin_in_lookback_time'] == True:
            ndec = 2
        else:
            ndec = 1
        slice_keys = self.slice_key_builder(ndecimal=ndec)

        # pdb.set_trace()

        for i in range(self.nz):
            z_slice = slice_keys[i]
            z_suf = 'z_' + self.z_keys[i]
            if self.params['bootstrap'] == True:
                for k in np.arange(self.nboots) + int(self.params['boot0']):
                    if self.params['bins']['stack_all_z_at_once'] == True:
                        filename_boots = 'simstack_flux_densities_' + self.params['io'][
                            'shortname'] + '_all_z' + '_boot_' + str(k) + '.p'
                    else:
                        filename_boots = 'simstack_flux_densities_' + self.params['io'][
                            'shortname'] + '_' + z_slice + '_boot_' + str(k) + '.p'

                    if os.path.exists(self.path + filename_boots):
                        bootstack = pickle.load(open(self.path + filename_boots, "rb"))
                        if self.params['save_bin_ids'] == True:
                            for bbk in bootstack[0].keys():
                                self.bin_ids[bbk + '_' + str(k)] = bootstack[0][bbk]
                        for wv in range(self.nw):
                            # pdb.set_trace1()
                            if self.params['save_bin_ids'] == True:
                                try:
                                    single_wv_stacks = bootstack[1][z_slice][self.maps[wv]]
                                except:
                                    single_wv_stacks = bootstack[1][self.maps[wv]]
                            else:
                                try:
                                    single_wv_stacks = bootstack[z_slice][self.maps[wv]]
                                except:
                                    single_wv_stacks = bootstack[self.maps[wv]]
                            for j in range(self.nm):
                                m_suf = 'm_' + self.m_keys[j]
                                for p in range(self.npops):
                                    p_suf = self.pops[p]
                                    key = clean_args(z_suf + '__' + m_suf + '_' + p_suf)
                                    # pdb.set_trace()
                                    try:
                                        bootstrap_fluxes[wv, i, j, p, k] = single_wv_stacks[key].value
                                        bootstrap_errors[wv, i, j, p, k] = single_wv_stacks[key].stderr
                                        bootstrap_intensities[wv, i, j, p, k] = single_wv_stacks[key].value * (
                                        self.fqs[wv]) * 1e-26 * 1e9
                                    except:
                                        bootstrap_fluxes[wv, i, j, p, k] = single_wv_stacks[key]['value']
                                        bootstrap_errors[wv, i, j, p, k] = single_wv_stacks[key]['stderr']
                                        bootstrap_intensities[wv, i, j, p, k] = single_wv_stacks[key]['value'] * (
                                        self.fqs[wv]) * 1e-26 * 1e9

                self.bootstrap_flux_array = bootstrap_fluxes
                self.bootstrap_error_array = bootstrap_errors
                self.bootstrap_nuInu_array = bootstrap_intensities
            else:
                if self.params['bins']['stack_all_z_at_once'] == True:
                    filename_stacks = 'simstack_flux_densities_' + self.params['io']['shortname'] + '_all_z' + '.p'
                else:
                    filename_stacks = 'simstack_flux_densities_' + self.params['io']['shortname'] + '_' + z_slice + '.p'
                if os.path.exists(self.path + filename_stacks):
                    simstack = pickle.load(open(self.path + filename_stacks, "rb"))
                    # pdb.set_trace()
                    for ssk in simstack[0]:
                        self.bin_ids[ssk] = simstack[0][ssk]
                    for wv in range(self.nw):
                        try:
                            single_wv_stacks = simstack[1][z_slice][self.maps[wv]]
                        except:
                            single_wv_stacks = simstack[1][self.maps[wv]]
                        for j in range(self.nm):
                            m_suf = 'm_' + self.m_keys[j]
                            for p in range(self.npops):
                                p_suf = self.pops[p]
                                key = clean_args(z_suf + '__' + m_suf + '_' + p_suf)
                                try:
                                    stacked_fluxes[wv, i, j, p] = single_wv_stacks[key].value
                                    try:
                                        stacked_errors[wv, i, j, p] = single_wv_stacks[key].psnerr
                                    except:
                                        stacked_errors[wv, i, j, p] = single_wv_stacks[key].stderr
                                    stacked_intensities[wv, i, j, p] = single_wv_stacks[key].value * (
                                                self.fqs[wv] * 1e9) * 1e-26 * 1e9
                                except:
                                    stacked_fluxes[wv, i, j, p] = single_wv_stacks[key]['value']
                                    try:
                                        stacked_errors[wv, i, j, p] = single_wv_stacks[key]['psnerr']
                                    except:
                                        stacked_errors[wv, i, j, p] = single_wv_stacks[key]['stderr']
                                    stacked_intensities[wv, i, j, p] = single_wv_stacks[key]['value'] * (
                                                self.fqs[wv] * 1e9) * 1e-26 * 1e9

                self.simstack_flux_array = stacked_fluxes
                self.simstack_error_array = stacked_errors
                self.simstack_nuInu_array = stacked_intensities

    def is_bootstrap(self, config):
        return config['bootstrap']

    def slice_key_builder(self, ndecimal=2):

        decimal_pre_lo = '{:.' + str(ndecimal) + 'f}'
        decimal_pre_hi = '{:.' + str(ndecimal) + 'f}'

        if self.params['bins']['bin_in_lookback_time']:
            z_nodes = self.params['bins']['t_nodes']
        else:
            z_nodes = self.params['bins']['z_nodes']
        nz = len(z_nodes) - 1

        slice_key = [str(decimal_pre_lo.format(z_nodes[i])) + '-' + str(decimal_pre_hi.format(z_nodes[i + 1])) for i in
                     range(nz)]
        if (slice_key[0] == '0.0-0.5') & (z_nodes[0] == 0.01):
            slice_key[0] = '0.01-0.5'
        # return [str(decimal_pre_lo.format(z_nodes[i]))+ '-' +str(decimal_pre_hi.format(z_nodes[i+1])) for i in range(nz)]
        return slice_key

    def m_z_key_builder(self, ndecimal=2):

        z_suf = []
        m_suf = []

        decimal_pre = '{:.' + str(ndecimal) + 'f}'

        for i in range(self.nz):
            z_suf.append(decimal_pre.format(self.params['bins']['z_nodes'][i]) + '-' + decimal_pre.format(
                self.params['bins']['z_nodes'][i + 1]))

        for j in range(self.nm):
            m_suf.append(decimal_pre.format(self.params['bins']['m_nodes'][j]) + '-' + decimal_pre.format(
                self.params['bins']['m_nodes'][j + 1]))

        return [z_suf, m_suf]


def measure_cib(stacked_object, area_deg=1.62, tcib=False):
    '''
	Sums the contribution from sources (in each bin) to the CIB at each wavelength.
	If tcib == True, output is sum of all bins at each wavelength.
	'''
    if area_deg == 1.62:
        print('defaulting to uVista/COSMOS area of 1.62deg2')
    area_sr = area_deg * (3.1415926535 / 180.) ** 2
    cib = np.zeros(np.shape(stacked_object.simstack_nuInu_array))
    for iwv in range(stacked_object.nw):
        for i in range(stacked_object.nz):
            zn = stacked_object.z_nodes[i:i + 2]
            z_suf = '{:.2f}'.format(zn[0]) + '-' + '{:.2f}'.format(zn[1])
            for j in range(stacked_object.nm):
                mn = stacked_object.m_nodes[j:j + 2]
                m_suf = '{:.2f}'.format(mn[0]) + '-' + '{:.2f}'.format(mn[1])
                for p in range(stacked_object.npops):
                    arg = clean_args('z_' + z_suf + '__m_' + m_suf + '_' + stacked_object.pops[p])
                    ng = len(stacked_object.bin_ids[arg])
                    cib[iwv, i, j, p] += 1e-9 * float(ng) / area_sr * stacked_object.simstack_nuInu_array[iwv, i, j, p]
    if tcib == True:
        return np.sum(np.sum(np.sum(cib, axis=1), axis=1), axis=1)
    else:
        return cib


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
