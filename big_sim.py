import numpy as np
import pymaster as nmt
import methods

ell, TT, TE, EE, BB, PP = np.loadtxt('cl.txt', unpack=True)
input_cl = np.array([TT, TE, EE, BB])
input_cl /= ell*(ell+1)/2/np.pi 
for c in input_cl: c[0] = 0

nside = 128
lmax = 3*nside-1
npix = 12*nside**2
rate = 10 #hz

pixels_lr, pixels_ud = methods.create_pixels()
TT = input_cl[0]
mask = np.zeros(npix)
mask[pixels_lr] = 1
mask_apo = nmt.mask_apodization(mask, 2.5, apotype="Smooth")    

ells, cl_fb = methods.sim_cl2cl(TT, (pixels_lr, pixels_ud), rate, n_sims=100, n_obs=100, replace=False)
np.save('ells', ells)
np.save('cl_fb', cl_fb)

ells, cl_ft = methods.sim_cl2cl(np.ones(lmax+1), (pixels_lr, pixels_ud), rate, n_sims=100, n_obs=100, noise_params=None, replace=False)
np.save('cl_ft', cl_ft)

ells, cl_replace = methods.sim_cl2cl(TT, (pixels_lr, pixels_ud), rate, n_sims=100, n_obs=100, replace=True)
np.save('cl_replace', cl_replace)

print('YEE HAW')