import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mylib
import pymaster as nmt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ground-P-noise-props',nargs='+', type=float)
parser.add_argument('--sat-T-noise-props',nargs='+', type=float)
parser.add_argument('--sat-P-noise-props',nargs='+', type=float)
parser.add_argument('--noBB', action='store_true')
parser.add_argument('--outname')
args = parser.parse_args()

nside = 512
npix = 12*nside**2
lmax = 3*nside-1
bin_size = 64
n_sims = 300

ell_input, TT, TE, EE, BB, PP = np.loadtxt('cl.txt', unpack=True)
input_cls = np.array([TT, EE, BB, TE]) 
input_cls /= ell_input*(ell_input+1)/2/np.pi 
for c in input_cls: c[0] = 0 
input_cls = input_cls[:,:lmax+1]
if args.noBB:
    input_cls[2] = np.zeros(lmax+1)
ell_input = ell_input[:lmax+1]
c2d = ell_input*(ell_input+1)/2/np.pi

Nl_P = mylib.get_Nl(args.ground_P_noise_props, lmax)
ground_noise_cls = np.array([np.zeros(lmax+1), Nl_P, Nl_P, np.zeros(lmax+1)])

Nl_T = mylib.get_Nl(args.sat_T_noise_props, lmax)
Nl_P = mylib.get_Nl(args.sat_P_noise_props, lmax)

sat_noise_cls = np.array([Nl_T, Nl_P, Nl_P, np.zeros(lmax+1)])

mask = mylib.get_mask(nside)
mask_apo = nmt.mask_apodization(mask, 6, apotype='C2')
fsky = len(mask_apo[(mask_apo!=0)])/npix
print(fsky *100)

sigmab = hp.nside2resol(nside)
fwhm = mylib.sigma2fwhm(sigmab)
bl = mylib.get_bl(nside)
b = nmt.NmtBin.from_nside_linear(nside, bin_size)
ells = b.get_effective_ells()
c2db = ells * (ells+1) /2/np.pi

w_KS = nmt.NmtWorkspace()
f_KS = nmt.NmtField(mask_apo, np.empty((2, npix)), beam=bl, purify_b=True)
w_KS.compute_coupling_matrix(f_KS, f_KS, b)

w_PCL = nmt.NmtWorkspace()
f_PCL = nmt.NmtField(mask_apo, np.empty((2, npix)), beam=bl, purify_b=False)
w_PCL.compute_coupling_matrix(f_PCL, f_PCL, b)

w_comb = nmt.NmtWorkspace()
f_comb = nmt.NmtField(mask_apo, np.empty((1, npix)), beam=bl)
w_comb.compute_coupling_matrix(f_comb, f_comb, b)

mll = w_comb.get_coupling_matrix()
masked_ground_noise_cls = mll @ ground_noise_cls[2]
masked_ground_noise_cls[0:2] = 0
masked_ground_noise_cls_pol = np.array([mll@ground_noise_cls[1], np.zeros(lmax+1), np.zeros(lmax+1), mll@ground_noise_cls[2]])
masked_ground_noise_cls_pol[:,0:2] = 0

cl_KS = np.empty((n_sims, len(ells)))
cl_PCL = np.empty((n_sims, len(ells)))
cl_comb = np.empty((n_sims, len(ells)))

for i in tqdm(range(n_sims)):
    input_map = hp.synfast(input_cls, nside, fwhm=fwhm, new=True)
    ground_noise = hp.synfast(ground_noise_cls, nside, new=True)
    sat_noise = hp.synfast(sat_noise_cls, nside, new=True)

    ground_map = input_map + ground_noise
    sat_map = input_map + sat_noise
    wienered_sat_map = mylib.wiener_filter(sat_map, input_cls, sat_noise_cls)
    comb_map = mask*ground_map + (1-mask)*wienered_sat_map
    B_map = hp.alm2map(hp.map2alm(comb_map)[2], nside)

    f_KS = nmt.NmtField(mask_apo, [ground_map[1], ground_map[2]], purify_b=True)
    f_PCL = nmt.NmtField(mask_apo, [ground_map[1], ground_map[2]], purify_b=False)
    f_comb = nmt.NmtField(mask_apo, [B_map])

    cl_KS[i] = nmt.compute_full_master(f_KS, f_KS, b, masked_ground_noise_cls_pol, workspace=w_KS)[3]
    cl_PCL[i] = nmt.compute_full_master(f_PCL, f_PCL, b, masked_ground_noise_cls_pol, workspace=w_PCL)[3]
    cl_comb[i] = nmt.compute_full_master(f_comb, f_comb, b, [masked_ground_noise_cls], workspace=w_comb)[0]
    
mean_KS = c2db*np.mean(cl_KS, axis=0)
std_KS = c2db*np.std(cl_KS, axis=0)

mean_PCL = c2db*np.mean(cl_PCL, axis=0)
std_PCL = c2db*np.std(cl_PCL, axis=0)

mean_comb = c2db*np.mean(cl_comb, axis=0)
std_comb = c2db*np.std(cl_comb, axis=0)

bpw = w_PCL.get_bandpower_windows()[3,:,3]
w2 = np.sum(mask_apo**2)/np.sum(mask)
w4 = np.sum(mask_apo**4)/np.sum(mask)
nu_l = (2*np.arange(lmax+1)+1)*fsky*w2**2/w4*bin_size
knox = (input_cls[2] + ground_noise_cls[2]/bl**2) * np.sqrt(2/nu_l)
# knox = bpw @ (((input_cls[2] + ground_noise_cls[2]/bl**2)) * np.sqrt(2/nu_l))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].errorbar(ells, mean_KS, yerr=std_KS, fmt='.', label='KS')
axes[0].errorbar(ells, mean_PCL, yerr=std_PCL, fmt='.', label='PCL')
axes[0].errorbar(ells, mean_comb, yerr=std_comb, fmt='.', label='comb')
axes[0].plot(c2d*masked_ground_noise_cls, linestyle='dashed', label=f'masked ground BB noise {args.ground_P_noise_props}')
axes[0].plot(c2d*sat_noise_cls[0], linestyle='dashed', label=f'sat T noise {args.sat_T_noise_props}')
axes[0].plot(c2d*sat_noise_cls[1], linestyle='dashed', label=f'sat P noise {args.sat_P_noise_props}')
axes[0].plot(c2d*input_cls[2], label='input BB')
axes[0].set_title('BB')

axes[1].plot(ells, std_KS, marker='.', label='KS')
axes[1].plot(ells, std_PCL, marker='.', label='PCL')
axes[1].plot(ells, std_comb, marker='.',label='comb')
axes[1].plot(c2d*knox, label='knox')
axes[1].set_title('std(BB)')

for ax in axes.flatten():
    ax.legend()
    ax.set_xlim([20,600])
    ax.loglog()
    ax.grid()
    ax.set_xlabel('ell')
    ax.set_ylabel('D_ell')

plt.savefig(f'{args.outname}')
