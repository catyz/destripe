import numpy as np
import pymaster as nmt
import methods
import sys 
import os

path = str(sys.argv[1])
n_sims = int(sys.argv[2])

if not os.path.exists(path):
    os.makedirs(path)

ell, TT, TE, EE, BB, PP = np.loadtxt('cl.txt', unpack=True)
input_cl = np.array([TT, TE, EE, BB])
input_cl /= ell*(ell+1)/2/np.pi 
for c in input_cl: c[0] = 0

nside = 128
lmax = 3*nside-1
npix = 12*nside**2

pixels = methods.create_pixels()
TT = input_cl[0][:lmax+1]
mask = np.zeros(npix)
mask[pixels[0]] = 1
mask_apo = nmt.mask_apodization(mask, 2.5, apotype="Smooth")    

fl = methods.sim_fl(TT, pixels, n_sims=n_sims, n_obs=100)
np.save(f'{path}/fl', fl)

ells, cl_fb = methods.sim_cl2cl(TT, pixels, fl, n_sims=n_sims, n_obs=100)
np.save(f'{path}/cl_fb', cl_fb)
np.save(f'{path}/ells', ells)

ells, cl_fr = methods.sim_cl2cl(TT, pixels, n_sims=n_sims, n_obs=100, replace=True)
np.save(f'{path}/cl_fr', cl_fr)

print('YEE HAW')