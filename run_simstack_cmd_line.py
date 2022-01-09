#!/usr/bin/env python

'''

Set Environment Variables

export MAPSPATH=$MAPSPATH/Users/marcoviero/data/Astronomy/maps/
export CATSPATH=$CATSPATH/Users/marcoviero/data/Astronomy/catalogs/
export PICKLESPATH=$PICKLESPATH/Users/marcoviero/data/Astronomy/pickles/

Setup New Virtual Environment
> conda create -n simstack python=3.9
> conda activate simstack

Install Packages
- matplotlib (> conda install matplotlib)
- seaborn (> conda install seaborn)
- numpy (> conda install numpy)
- pandas (> conda install pandas)
- astropy (> conda install astropy)
- lmfit (> conda install -c conda-forge lmfit)
- [jupyterlab, if you want to use notebooks]

To run from command line:
- First make this file executable (only needed once), e.g.:
> chmod +x run_simstack_cmd_line.py
- Run script:
> python run_simstack_cmd_line.py
'''

# Standard modules
import os
import time
import logging

# Modules within this package
from simstackwrapper import SimstackWrapper

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%d-%m %I:%M:%S %p')

    # Get parameters from the provided parameter file
    #param_file_path = sys.argv[1]
    param_file_path = os.path.join('examples', 'uvista.ini')

    # Instantiate SIMSTACK object
    simstack_object = SimstackWrapper(param_file_path)

    t0 = time.time()

    # Begin Stacking
    simstack_object.perform_simstack()

    # Rearrange results for plotting
    simstack_object.parse_results()

    # Save Results; they are stored in e.g., simstack_object.maps_dict['spire_plw']['stacked_flux_densities']
    simstack_object.save_stacked_fluxes(param_file_path)

    # Summarize timing
    t1 = time.time()
    tpass = t1 - t0

    logging.info("Done!")
    logging.info("")
    logging.info("Total time                        : {:.4f} minutes\n".format(tpass / 60.))

if __name__ == "__main__":
    main()
else:
    logging.info("Note: `mapit` module not being run as main executable.")
