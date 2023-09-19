#!/bin/sh

python filtering_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --outname 'filtering_sim/noBB_planck'

python filtering_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'filtering_sim/noBB_noiseless'

python filtering_sim.py --ground-P-noise-props 3.3 50 -3 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --outname 'filtering_sim/sosat_planck'

python filtering_sim.py --ground-P-noise-props 3.3 50 -3 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'filtering_sim/sosat_noiseless'

python filtering_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --outname 'filtering_sim/noiseless_planck' 

python filtering_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'filtering_sim/noiseless_noiseless'