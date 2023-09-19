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
bin_size = 20
n_sims = 500

ell_input, TT, TE, EE, BB, PP = np.loadtxt('cl.txt', unpack=True)
# input_cls = np.array([TT, EE, np.zeros_like(EE), TE]) #no BB
input_cls = np.array([TT, EE, BB, TE]) 
input_cls /= ell_input*(ell_input+1)/2/np.pi 
for c in input_cls: c[0] = 0 
input_cls = input_cls[:,:lmax+1]

if args.noBB:
    input_cls[2] = np.zeros(lmax+1)
ell_input = ell_input[:lmax+1]
c2d = ell_input*(ell_input+1)/2/np.pi

input_cls_onlyBB = np.copy(input_cls)
input_cls_onlyBB[0] = np.zeros(lmax+1)
input_cls_onlyBB[1] = np.zeros(lmax+1)
input_cls_onlyBB[3] = np.zeros(lmax+1)

#Baseline/goal SAT 145 GHz. l_knee = 50/25, n= 3.3/2.1
# ground_P_noise_props = [3, 0, 0]
Nl_P = mylib.get_Nl(args.ground_P_noise_props, lmax)
ground_noise_cls = np.array([np.zeros(lmax+1), Nl_P, Nl_P, np.zeros(lmax+1)])

#Planck Npipe2020 144ghz
# sat_T_noise_props = [25, 0, 0]
# sat_P_noise_props = [50, 0, 0]
Nl_T = mylib.get_Nl(args.sat_T_noise_props, lmax)
Nl_P = mylib.get_Nl(args.sat_P_noise_props, lmax)
sat_noise_cls = np.array([Nl_T, Nl_P, Nl_P, np.zeros(lmax+1)])

mask = mylib.get_mask(nside)
mask_apo = nmt.mask_apodization(mask, 6, apotype='C2')
fsky = len(mask_apo[(mask_apo!=0)])/npix
w2 = np.sum(mask_apo**2)/np.sum(mask)
w4 = np.sum(mask_apo**4)/np.sum(mask)
print(fsky *100)
pix = mylib.mask2pix(mask)

sigmab = hp.nside2resol(nside)
fwhm = mylib.sigma2fwhm(sigmab)
bl = mylib.get_bl(nside)
b = nmt.NmtBin.from_nside_linear(nside, bin_size)
ells = b.get_effective_ells()
c2db = ells * (ells+1) /2/np.pi
mll,_ = mylib.get_mll(mask_apo, nside, b, pol=False)

#BB Filter transfer
pcl_bb_vec = np.empty((n_sims, lmax+1))
# cl_input_vec = np.empty((n_sims, lmax+1))

for i in tqdm(range(n_sims)):
    np.random.seed(i+1000)
    input_map = hp.synfast(input_cls_onlyBB, nside, fwhm=fwhm, new=True)
    filtered_input_map = mask_apo * mylib.tod_filter(input_map, pix, deg=20)
    pcl_bb_vec[i] = hp.anafast(filtered_input_map)[2]
#     cl_input_vec[i] = hp.anafast(input_map)/bl**2
    
# cl_input = np.mean(cl_input_vec, axis=0)
pcl_bb = np.mean(pcl_bb_vec, axis=0)

if np.abs(pcl_bb).max() == 0:
    fl_bb = np.ones(lmax+1)
    fl_bb[0:2] = 0
else:
    fl_bb = mylib.get_fl(input_cls[2], pcl_bb, bl, mll)

w_KS = nmt.NmtWorkspace()
f_KS = nmt.NmtField(mask_apo, np.empty((2, npix)), beam=bl*np.sqrt(fl_bb), purify_b=True)
w_KS.compute_coupling_matrix(f_KS, f_KS, b)

w_comb = nmt.NmtWorkspace()
f_comb = nmt.NmtField(mask_apo, np.empty((1, npix)), beam=bl*np.sqrt(fl_bb))
w_comb.compute_coupling_matrix(f_comb, f_comb, b)

masked_ground_noise_cls = fl_bb*mll @ ground_noise_cls[2]
masked_ground_noise_cls_pol = np.array([fl_bb*mll@ground_noise_cls[1], np.zeros(lmax+1), np.zeros(lmax+1), fl_bb*mll@ground_noise_cls[2]])


#Knox formula
bpw = w_comb.get_bandpower_windows()[0,:,0]
nu_l = (2*np.arange(lmax+1)+1)*fsky*w2**2/w4*bin_size*fl_bb
knox = (input_cls[2] + ground_noise_cls[2]/bl**2) * np.sqrt(2/nu_l)
knox[0:2] = 0

cl_KS = np.empty((n_sims, len(ells)))
cl_restore_ext = np.empty((n_sims, len(ells)))
cl_KS_restore_patch = np.empty((n_sims, len(ells)))
cl_restore_both = np.empty((n_sims, len(ells)))

for i in tqdm(range(n_sims)):
    np.random.seed(i)
    input_map = hp.synfast(input_cls, nside, fwhm=fwhm, new=True)
    ground_noise = hp.synfast(ground_noise_cls, nside, new=True)
    sat_noise = hp.synfast(sat_noise_cls, nside, new=True)

    ground_map = input_map + ground_noise
    filtered_ground_map = mylib.tod_filter(ground_map, pix, deg=20)

    sat_map = input_map + sat_noise
    wienered_sat_map = mylib.wiener_filter(sat_map, input_cls, sat_noise_cls)
    restored_map_ext = mask*filtered_ground_map + (1-mask)*wienered_sat_map
    B_map_ext = hp.alm2map(hp.map2alm(restored_map_ext)[2], nside)
    
    lost_map = wienered_sat_map - mylib.tod_filter(wienered_sat_map, pix, deg=20)
    restored_map_both = filtered_ground_map + lost_map
    B_map_both = hp.alm2map(hp.map2alm(restored_map_both)[2], nside)
    
    f_KS = nmt.NmtField(mask_apo, mask*[filtered_ground_map[1], filtered_ground_map[2]], purify_b=True)
    f_restore_ext = nmt.NmtField(mask_apo, mask*[B_map_ext])
    f_KS_restore_patch = nmt.NmtField(mask_apo, mask*[restored_map_both[1], restored_map_both[2]], purify_b=True)
    f_restore_both = nmt.NmtField(mask_apo, mask*[B_map_both])

    cl_KS[i] = nmt.compute_full_master(f_KS, f_KS, b, masked_ground_noise_cls_pol, workspace=w_KS)[3]
    cl_restore_ext[i] = nmt.compute_full_master(f_restore_ext, f_restore_ext, b, [masked_ground_noise_cls], workspace=w_comb)[0]
    cl_KS_restore_patch[i] = nmt.compute_full_master(f_KS_restore_patch, f_KS_restore_patch, b, masked_ground_noise_cls_pol, workspace=w_KS)[3]
    cl_restore_both[i] = nmt.compute_full_master(f_restore_both, f_restore_both, b, [masked_ground_noise_cls], workspace=w_comb)[0]

mean_KS = c2db*np.mean(cl_KS, axis=0)
std_KS = c2db*np.std(cl_KS, axis=0)

mean_restore_ext = c2db*np.mean(cl_restore_ext, axis=0)
std_restore_ext = c2db*np.std(cl_restore_ext, axis=0)

mean_KS_restore_patch = c2db*np.mean(cl_KS_restore_patch, axis=0)
std_KS_restore_patch = c2db*np.std(cl_KS_restore_patch, axis=0)

mean_restore_both = c2db*np.mean(cl_restore_both, axis=0)
std_restore_both = c2db*np.std(cl_restore_both, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].errorbar(ells, mean_KS, yerr=std_KS, fmt='.', label='KS')
axes[0].errorbar(ells, mean_restore_ext, yerr=std_restore_ext, fmt='.', label='restore external')
# axes[0].errorbar(ells, mean_PCL, yerr=std_PCL, fmt='.', label='PCL')
axes[0].errorbar(ells, mean_KS_restore_patch, yerr=std_KS_restore_patch, fmt='.', label='KS + restore patch')
axes[0].errorbar(ells, mean_restore_both, yerr=std_restore_both, fmt='.', label='restore both')

axes[0].plot(c2d*masked_ground_noise_cls, linestyle='dashed', label=f'masked ground BB noise {args.ground_P_noise_props}')
axes[0].plot(c2d*sat_noise_cls[0], linestyle='dashed', label=f'sat T noise {args.sat_T_noise_props}')
axes[0].plot(c2d*sat_noise_cls[1], linestyle='dashed', label=f'sat P noise {args.sat_P_noise_props}')
axes[0].plot(c2d*input_cls[2], label='input BB')
axes[0].set_title('BB')

axes[1].plot(ells, std_KS, marker='.', label='KS')
axes[1].plot(ells, std_restore_ext, marker='.',label='restore external')
axes[1].plot(ells, std_KS_restore_patch, marker='.', label='KS + restore patch')
# axes[1].plot(ells, std_PCL, marker='.', label='PCL')
axes[1].plot(ells, std_restore_both, marker='.',label='restore both')
axes[1].plot(ells, c2db*(bpw@knox), label='knox')
axes[1].set_title('std(BB)')

for ax in axes.flatten():
    ax.legend()
    ax.set_xlim([10,600])
    ax.loglog()
    ax.grid()
    ax.set_xlabel('ell')
    ax.set_ylabel('D_ell')
    
plt.savefig(f'{args.outname}')