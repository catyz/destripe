#!/bin/sh

#White noise
python partial_sky_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 0 0 --sat-P-noise-props 50 0 0 --outname 'white_noise/noBB_planck'

python partial_sky_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'white_noise/noBB_noiseless'

python partial_sky_sim.py --ground-P-noise-props 10 0 0 --sat-T-noise-props 25 0 0 --sat-P-noise-props 50 0 0 --outname 'white_noise/10uK_planck'

python partial_sky_sim.py --ground-P-noise-props 10 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'white_noise/10uK_noiseless'

python partial_sky_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 0 0 --sat-P-noise-props 50 0 0 --outname 'white_noise/noiseless_planck' 

python partial_sky_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'white_noise/noiseless_noiseless'


#With 1/f
python partial_sky_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --outname 'red_noise/noBB_planck'

python partial_sky_sim.py --noBB --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'red_noise/noBB_noiseless'

python partial_sky_sim.py --ground-P-noise-props 10 50 -3 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --outname 'red_noise/10uK_planck'

python partial_sky_sim.py --ground-P-noise-props 10 50 -3 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'red_noise/10uK_noiseless'

python partial_sky_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 25 50 -1.8 --sat-P-noise-props 50 20 -1.4 --outname 'red_noise/noiseless_planck' 

python partial_sky_sim.py --ground-P-noise-props 0 0 0 --sat-T-noise-props 0 0 0 --sat-P-noise-props 0 0 0 --outname 'red_noise/noiseless_noiseless'


