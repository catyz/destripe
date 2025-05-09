#!/bin/sh

#polydeg 10
python filtering_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --poly-deg 10 --outname 'noBB_planck'

python filtering_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --poly-deg 10 --outname 'noBB_noiseless'

python filtering_sim.py --ground-P-noise-props 3.3 50 -3 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --poly-deg 10 --outname 'sosat_planck'

python filtering_sim.py --ground-P-noise-props 3.3 50 -3 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --poly-deg 10 --outname 'sosat_noiseless'

python filtering_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --poly-deg 10 --outname 'noiseless_planck' 

python filtering_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --poly-deg 10 --outname 'noiseless_noiseless'


#polydeg 20
python filtering_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --poly-deg 20 --outname 'noBB_planck'

python filtering_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --poly-deg 20 --outname 'noBB_noiseless'

python filtering_sim.py --ground-P-noise-props 3.3 50 -3 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --poly-deg 20 --outname 'sosat_planck'

python filtering_sim.py --ground-P-noise-props 3.3 50 -3 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --poly-deg 20 --outname 'sosat_noiseless'

python filtering_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --poly-deg 20 --outname 'noiseless_planck' 

python filtering_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --poly-deg 20 --outname 'noiseless_noiseless'