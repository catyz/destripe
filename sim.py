import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import methods
import pymaster as nmt
from importlib import reload
from tqdm import tqdm

reload(methods)
ell_input, TT, TE, EE, BB, PP = np.loadtxt('cl.txt', unpack=True)

nside = 128
lmax = 3*nside-1
npix = 12*nside**2
pixels = methods.create_pixels()
b = nmt.NmtBin.from_nside_linear(nside, 16)
ells = b.get_effective_ells()

input_cl = np.array([TT, EE, BB, TE])
input_cl /= ell_input*(ell_input+1)/2/np.pi 
for c in input_cl: c[0] = 0
input_cl = input_cl[:,:lmax+1]
ell_input = ell_input[:lmax+1]
    
input_cl = np.array([np.zeros(lmax+1), input_cl[1], np.zeros(lmax+1), np.zeros(lmax+1)]) #EE only
# input_cl = np.array([np.zeros(lmax+1), np.zeros(lmax+1), input_cl[2], np.zeros(lmax+1)]) #lensed r=0.1

mask = np.zeros(npix)
mask[pixels[0]] = 1
mask_apo = nmt.mask_apodization(mask, 5, apotype="Smooth") 
mll = methods.get_mll(mask_apo, nside)


n_sims = 100
fwhm_grd = 1

flat_cl = np.ones(lmax+1)
pcl = methods.sim_pcl(flat_cl, pixels, fwhm_grd=fwhm_grd, n_sims=n_sims, n_obs=1, map_seed_start=1000)
fl = methods.get_fl(flat_cl, pcl, hp.gauss_beam(np.deg2rad(fwhm_grd), lmax), mll, niter=3)
np.save('fl', fl)

# mean, std = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd,fl=fl, leakage=None, pure_b=False, n_sims=n_sims)
# np.save('pcl', (mean, std))

mean_p, std_p = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd,fl=fl, leakage=None, pure_b=True, n_sims=n_sims)
np.save('pure', (mean_p, std_p))

# mean_r, std_r = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd, fl=None, pure_b=True, n_sims=n_sims, replace=True)
# np.save('replace', (mean_r, std_r))

mean_c, std_c = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd, fwhm_sat = 1.0, fl=None, pure_b=True, n_sims=n_sims, combine=True)
np.save('combine_1p0', (mean_c, std_c))

mean_c, std_c = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd, fwhm_sat = 1.5, fl=None, pure_b=True, n_sims=n_sims, combine=True)
np.save('combine_1p5', (mean_c, std_c))

# mean_c, std_c = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd, fwhm_sat = 2.0, fl=None, pure_b=True, n_sims=n_sims, combine=True)
# np.save('combine_2p0', (mean_c, std_c))

mean_f, std_f = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd, fwhm_sat = 1.0, fl=None, pure_b=False, n_sims=n_sims, combine=True, fill=True)
np.save('full_1p0', (mean_f, std_f))

mean_f, std_f = methods.run_sim(input_cl, pixels,nside, mask, b, fwhm_grd=fwhm_grd, fwhm_sat = 1.5, fl=None, pure_b=False, n_sims=n_sims, combine=True, fill=True)
np.save('full_1p5', (mean_f, std_f))

print('yee haw')
