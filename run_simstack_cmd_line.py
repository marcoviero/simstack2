#!/usr/bin/env python

'''

Required Environment Variables

export MAPSPATH=$MAPSPATH/Users/marcoviero/data/Astronomy/maps/
export CATSPATH=$CATSPATH/Users/marcoviero/data/Astronomy/catalogs/
export PICKLESPATH=$PICKLESPATH/Users/marcoviero/data/Astronomy/pickles/

Required Packages
-
- lmfit
'''

# Standard modules
import pdb
import os
import os.path
import sys
import shutil
import time
import logging
import pickle

# Modules within this package
from simstack import SimstackWrapper

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%d-%m %I:%M:%S %p')

    # Get parameters from the provided parameter file
    #param_file_path = sys.argv[1]
    param_file_path = os.path.join('examples','example.ini')#sys.argv[1]

    # Instantiate SIMSTACK object
    simstack_object = SimstackWrapper(param_file_path)

    t0 = time.time()

    # Begin Stacking
    simstack_object.perform_simstack()

    simstack_object.parse_results()
    # Save Results; they are stored in e.g., simstack_object.maps_dict['spire_plw']['stacked_flux_densities']
    #save_stacked_fluxes(stacked_flux_densities, params, out_file_path, out_file_suffix, IDs=bin_ids)
    # pdb.set_trace()

    # Save Parameter file in folder
    #save_paramfile(simstack_object.config_dict)

    # Summarize timing
    t1 = time.time()
    tpass = t1 - t0

    logging.info("Done!")
    logging.info("")
    logging.info("Total time                        : {:.4f} minutes\n".format(tpass / 60.))

    pdb.set_trace()

def save_stacked_fluxes(stacked_fluxes, params, out_file_path, out_file_suffix, IDs=None):
    fpath = "%s/%s_%s%s.p" % (
        out_file_path, params['io']['flux_densities_filename'], params['io']['shortname'], out_file_suffix)
    print('pickling to ' + fpath)
    if not os.path.exists(out_file_path): os.makedirs(out_file_path)

    if IDs == None:
        pickle.dump(stacked_fluxes, open(fpath, "wb"))  # , protocol=2 )
    else:
        pickle.dump([IDs, stacked_fluxes], open(fpath, "wb"))  # , protocol=2 )

def save_paramfile(params):
    fp_in = params['io']['param_file_path']
    if params['bootstrap'] == True:
        outdir = params['io']['output_folder'] + '/bootstrapped_fluxes/' + params['io']['shortname']
    else:
        outdir = params['io']['output_folder'] + '/simstack_fluxes/' + params['io']['shortname']
    print('writing parameter file to ' + outdir)
    if not os.path.exists(outdir): os.makedirs(outdir)
    fname = os.path.basename(fp_in)
    fp_out = os.path.join(outdir, fname)

    logging.info("Copying parameter file...")
    logging.info("  FROM : {}".format(fp_in))
    logging.info("    TO : {}".format(fp_out))
    logging.info("")

    shutil.copyfile(fp_in, fp_out)


if __name__ == "__main__":
    main()
else:
    logging.info("Note: `mapit` module not being run as main executable.")
