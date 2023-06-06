import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
from tqdm import tqdm
import scipy.signal as signal

def get_mask(nside):
    msk = np.zeros(hp.nside2npix(nside))
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    ph[np.where(ph > np.pi)[0]] -= 2 * np.pi
    msk[np.where((th < 2.63) & (th > 1.86) &
                 (ph > -np.pi / 4) & (ph < np.pi / 4))[0]] = 1.
    return msk

def depth2nl(map_depth, beam_fwhm, lmax):
    ell = np.arange(lmax)
    return map_depth**2 * np.exp(ell*(ell+1) * beam_fwhm **2 / 8/np.log(2))

def sigma2fwhm(sigma):
    return (8*np.log(2))**0.5 * sigma

def fwhm2sigma(fwhm):
    return (8*np.log(2))**-0.5 * fwhm

def almxfl(alm, fl):
    #For polarized alm, for which hp is too dumb to handle
    alm_corrected = np.empty_like(alm)
    for i in range(3):
        alm_corrected[i] = hp.almxfl(alm[i], fl)
    
    return alm_corrected

def create_pixels():
    #Hardcoded for nside 128, lol
    row_width = 200 #pixels
    offset = 312
    start = 35000
    nrows = 200
    
    pixels_lr = np.array([np.arange(start+i*(offset+row_width), start+row_width+i*(offset+row_width)) for i in range(nrows)])
    pixels_ud = np.copy(pixels_lr.T)
    
    for pixels in (pixels_lr, pixels_ud):
        pixels[::2] = pixels[::2, ::-1] #flip even rows
        
    return np.concatenate(pixels_lr), np.concatenate(pixels_ud)

def PT(y, pixels, npix):
    return np.bincount(pixels, y, minlength=npix)

def PTP(pixels, npix):
    return np.bincount(pixels, minlength=npix)

def generate_noise(nsamp, dt, fknee, alpha, sigma, seed=None):
    freq = np.abs(np.fft.fftfreq(nsamp, dt))
    noise_spec = (1+(np.maximum(freq,freq[1])/fknee)**-alpha)*sigma**2
    rand = np.fft.fft(np.random.default_rng(seed).standard_normal(nsamp))
    return np.fft.ifft(rand * noise_spec**0.5).real
    
def subscan_polyfilter(tod, n_sub=400, deg=10):
    times = np.arange(len(tod))
    subscans = np.array_split(np.copy(tod), n_sub)
    subscans_times = np.array_split(times, n_sub)
    for time, scan in zip(subscans_times, subscans):
        poly = np.polynomial.polynomial.Polynomial.fit(time, scan, deg)
        scan -= poly(time)
    return np.concatenate(subscans)

def high_pass(tod, freq=0.1, order=3, rate=10):
    sos = signal.butter(order, freq, 'hp', fs=rate, output='sos')
    return signal.sosfilt(sos, tod)

def map2map(input_map, pixels, noise_params, noise_seed=None, subscan_poly_deg=10, replace=False, plot=False):
    nside = 128
    npix = 12*nside**2
    nsamp = len(pixels)    

    signal = input_map[pixels]
    noise = np.zeros_like(signal)
    
    if noise_params is not None:
        rate, fknee, alpha, sigma = noise_params
        noise = generate_noise(nsamp, 1/rate, fknee, alpha, sigma, noise_seed)
        
    tod = signal + noise
    filtered_tod = subscan_polyfilter(tod, deg=subscan_poly_deg)
#     filtered_tod = high_pass(tod)

    
    if replace:
#         m_planck = hp.smoothing(input_map, fwhm=np.deg2rad(0)) 
        m_planck = input_map
        planck_tod = m_planck[pixels]
        filtered_planck_tod = subscan_polyfilter(planck_tod, deg=subscan_poly_deg)
#         filtered_planck_tod = high_pass(planck_tod)
        lost_signal = planck_tod - filtered_planck_tod
        filtered_tod += lost_signal
        
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(15, 5))
        ax[0].plot(tod, label='unfiltered', alpha=0.75)
        ax[0].plot(filtered_tod, label='filtered', alpha=0.75)
        ax[0].plot(signal, label='signal', alpha=0.75)
                
        freqs = np.fft.rfftfreq(nsamp, 1/rate)
        ax[1].loglog(freqs, np.abs(np.fft.rfft(tod)), label='unfiltered', alpha=0.75)
        ax[1].plot(freqs, np.abs(np.fft.rfft(filtered_tod)), label='filtered', alpha=0.75)
        ax[1].plot(freqs, np.abs(np.fft.rfft(signal)), label='signal', alpha=0.75)

        for a in ax:
            a.legend()
            a.grid()
            
    output_map = PT(filtered_tod, pixels, npix)/PTP(pixels, npix)
    output_map[np.isnan(output_map)] = 0
    
    return output_map

def coadd_split(input_map, pixels, n_obs, I_noise_params, P_noise_params, noise_seed_start=0, replace=False):
    coadd_map1 = np.zeros_like(input_map)
    coadd_map2 = np.zeros_like(input_map)
    
    for i in range(n_obs):
        noise_seed = noise_seed_start + 1000*i
        
        if len(input_map) == 3:

            for j in range(3):
                if j == 0: 
                    noise_params = I_noise_params
                    subscan_poly_deg = 10
                else: 
                    noise_params = P_noise_params
                    subscan_poly_deg = 10

                if len(input_map) == 3:
                    coadd_map1[j] += map2map(input_map[j], pixels[0], noise_params, noise_seed, subscan_poly_deg, replace=replace, plot=False)
                    coadd_map1[j] += map2map(input_map[j], pixels[1], noise_params, noise_seed+1, subscan_poly_deg, replace=replace, plot=False)

                    coadd_map2[j] += map2map(input_map[j], pixels[0], noise_params, noise_seed+2, subscan_poly_deg, replace=replace, plot=False)
                    coadd_map2[j] += map2map(input_map[j], pixels[1], noise_params, noise_seed+3, subscan_poly_deg, replace=replace, plot=False)
                
        else:
            noise_params = I_noise_params
            subscan_poly_deg = 10
            
            coadd_map1 += map2map(input_map, pixels[0], noise_params, noise_seed, subscan_poly_deg, replace=replace, plot=False)
            coadd_map1 += map2map(input_map, pixels[1], noise_params, noise_seed+1, subscan_poly_deg, replace=replace, plot=False)

            coadd_map2 += map2map(input_map, pixels[0], noise_params, noise_seed+2, subscan_poly_deg, replace=replace, plot=False)
            coadd_map2 += map2map(input_map, pixels[1], noise_params, noise_seed+3, subscan_poly_deg, replace=replace, plot=False)

    coadd_map1 /= 2*n_obs
    coadd_map2 /= 2*n_obs
    
    return coadd_map1, coadd_map2

def compute_master(f_a, f_b, wsp, leakage=None):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_coupled_leak = np.copy(cl_coupled)
    if leakage is not None:
        cl_coupled[3] -= leakage
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

def get_mll(mask_apo, nside):
    w = nmt.NmtWorkspace()
    b = nmt.NmtBin.from_nside_linear(nside, 16)
    f = nmt.NmtField(mask_apo, [np.ones(12*nside**2)])
    w.compute_coupling_matrix(f, f, b)
    return w.get_coupling_matrix()

def get_bl(nside):
    lmax = 3*nside-1
    sigmab = hp.nside2resol(nside)
    fwhm = (8*np.log(2))**0.5 * sigmab
    return hp.gauss_beam(fwhm, lmax)

def fl_itr(fl, pcl, mll, bcl):
    return fl + (pcl - fl * bcl) / bcl

def get_fl(input_cl, pcl, bl, mll, niter=3):
    bcl = mll @ (input_cl * bl**2)
    fl_i = pcl / bcl
    for j in range(niter):
        fl_i = fl_itr(fl_i, pcl, mll, bcl)
    return fl_i

def get_P_bl(ell_centers, nside):
    width = np.mean(np.diff(ell_centers))
    P_bl = np.zeros((len(ell_centers),3*nside))
    for i, ell_center in enumerate(ell_centers):
        l_low = int(ell_center-width/2)
        l_high = int(ell_center+width/2)
        P_bl[i][l_low: l_high] = 1/(l_high - l_low)
        
    return P_bl

def get_Q_lb(ell_centers, nside):
    width = np.mean(np.diff(ell_centers))
    Q_bl = np.zeros((len(ell_centers), 3*nside))
    for i, ell_center in enumerate(ell_centers):
        l_low = int(ell_center-width/2)
        l_high = int(ell_center+width/2)
        Q_bl[i][l_low: l_high] = 1
        
    return Q_bl.T

# def map2cl(ells, mask_apo, fl, m1, m2):    
#     nside = hp.get_nside(mask_apo)
#     bl = get_bl(nside)
#     mll = get_mll(mask_apo, nside)
#     P_bl = get_P_bl(ells, nside)
#     Q_lb = get_Q_lb(ells, nside)
    
#     pcl = hp.anafast(mask_apo*m1, mask_apo*m2)[:4]
#     debiased_cl = np.zeros((4, len(ells)))
    
#     for i in range(4):
#         if fl is not None:
#             K_bb_inv = np.linalg.inv(P_bl @ mll * fl[i] * bl**2 @ Q_lb)
#         else:
#             K_bb_inv = np.linalg.inv(P_bl @ mll * bl**2 @ Q_lb)
    
#         debiased_cl[i] = K_bb_inv @ P_bl @ pcl[i]
        
#     return debiased_cl
    
def sim_pcl(input_cl, pixels, fwhm_grd=1, n_sims=1, n_obs=1, map_seed_start=0):
    nside=128   
    lmax = 3*nside-1
    
    mask = np.zeros(12*nside**2)
    mask[pixels[0]] = 1
    mask_apo = nmt.mask_apodization(mask, 5, apotype="Smooth")
    
    if len(input_cl) != 4: #hardcoded kek
        pcl = np.zeros((lmax+1))
        for i in tqdm(range(n_sims)):
            if map_seed_start is not None:
                np.random.seed(map_seed_start + i)
            input_map = hp.synfast(input_cl, nside, fwhm=np.deg2rad(fwhm_grd))
            coadd_map1, coadd_map2 = coadd_split(input_map, pixels, n_obs, I_noise_params=None, P_noise_params=None, noise_seed_start=0, replace=False)
            pcl += hp.anafast(mask_apo*coadd_map1, mask_apo*coadd_map2)
        pcl /= n_sims
        return pcl
        
    else:
        print('IQU mode')
        pcl = np.zeros((n_sims, 4, lmax+1))
        for i in tqdm(range(n_sims)):
            if map_seed_start is not None:
                np.random.seed(map_seed_start + i)
            input_map = hp.synfast(input_cl, nside, fwhm=np.deg2rad(fwhm_grd), new=True)
            coadd_map1, coadd_map2 = coadd_split(input_map, pixels, n_obs, I_noise_params=None, P_noise_params=None, noise_seed_start=0, replace=False)
            pcl[i] = hp.anafast(mask_apo*coadd_map1, mask_apo*coadd_map2)[:4]
        
        return np.mean(pcl, axis=0) 

# def sim_cl2cl(input_cl, pixels, fl=None, n_sims=1, n_obs=1, I_noise_params=(10,0.5,2,100), P_noise_params=(10,0.1,0.1,150), noise_seed_start=0, map_seed_start=0, bin_size=16, replace=False):
#     nside=128        
#     lmax = 3*nside-1
#     sigmab = hp.nside2resol(nside)
#     bl = get_bl(nside)
#     ell_bin_centers = nmt.NmtBin.from_nside_linear(nside, bin_size).get_effective_ells()
#     mask = np.zeros(12*nside**2)
#     mask[pixels[0]] = 1
#     mask_apo = nmt.mask_apodization(mask, 5, apotype="Smooth")
    
#     cl = np.zeros((n_sims, input_cl.shape[0], len(ell_bin_centers)))
    
#     for i in tqdm(range(n_sims)):
#         if map_seed_start is not None:
#             np.random.seed(map_seed_start + i)
#         input_map = hp.synfast(input_cl, nside, sigma=sigmab, new=True)        
#         coadd_map1, coadd_map2 = coadd_split(input_map, pixels, n_obs, I_noise_params, P_noise_params, noise_seed_start, replace)
        
#         debiased_cross_cl = map2cl(ell_bin_centers, mask_apo, fl, coadd_map1, coadd_map2)
#         cl[i] = debiased_cross_cl
    
#     return ell_bin_centers, cl

def run_sim(input_cl, pixels, nside, mask, b, fwhm_grd=1, fwhm_sat=1.5, fl=None, leakage=None, pure_b=True, replace=False, combine=False, fill=False, n_sims=10, map_seed_start=123):
    lmax = 3*nside-1
    w = nmt.NmtWorkspace()
    ells = b.get_effective_ells()
    mask_apo = nmt.mask_apodization(mask, 5, apotype="Smooth") 
    
    if combine is True:
        bl = hp.gauss_beam(np.deg2rad(fwhm_sat), lmax)
        f = nmt.NmtField(mask_apo, [np.empty(12*nside**2), np.empty(12*nside**2)], beam=bl, purify_b=pure_b)
        if fill is True:
            f = nmt.NmtField(np.ones_like(mask), [np.empty(12*nside**2), np.empty(12*nside**2)], beam=bl, purify_b=False)
    else:
        bl = hp.gauss_beam(np.deg2rad(fwhm_grd), lmax)        
        f = nmt.NmtField(mask_apo, [np.empty(12*nside**2), np.empty(12*nside**2)], beam=bl, purify_b=pure_b)
        
    w.compute_coupling_matrix(f, f, b)
    
    if fl is not None:
        fl4 = np.interp(np.arange(0,lmax+1, 0.25), np.arange(lmax+1), fl)
        mll = w.get_coupling_matrix()
        w.update_coupling_matrix(mll * fl4)
    
    Cls = np.zeros((n_sims, 4, len(ells)))
    
    for i in tqdm(range(n_sims)):
        np.random.seed(map_seed_start + i)
        if combine is True:
            input_map = hp.synfast(input_cl, nside, fwhm=np.deg2rad(fwhm_sat), new=True)   
        else:
            input_map = hp.synfast(input_cl, nside, fwhm=np.deg2rad(fwhm_grd), new=True)   
            
        coadd_map1, coadd_map2 = coadd_split(input_map, pixels, n_obs=1, I_noise_params=None, P_noise_params=None, replace=replace)
        
        if combine is True:
            lost_map1 = mask*input_map - coadd_map1
            lost_map2 = mask*input_map - coadd_map2
            
#             alml1 = hp.map2alm(lost_map1)
#             alml2 = hp.map2alm(lost_map2)
                        
#             alml1 = almxfl(alml1, 1/bl_diff)
#             alml2 = almxfl(alml2, 1/bl_diff)

#             b_diff_fwhm = np.sqrt(fwhm_sat**2 - fwhm_grd**2)
#             alm1 = hp.map2alm(hp.smoothing(coadd_map1, fwhm=b_diff_fwhm))
#             alm2 = hp.map2alm(hp.smoothing(coadd_map2, fwhm=b_diff_fwhm))

#             alm1 = hp.map2alm(coadd_map1)
#             alm2 = hp.map2alm(coadd_map2)
            
#             coadd_map1 = mask*hp.alm2map(alm1 + alml1, nside)
#             coadd_map2 = mask*hp.alm2map(alm2 + alml2, nside)
            coadd_map1 += lost_map1
            coadd_map2 += lost_map2
        
            #For noiseless only
            assert np.abs(coadd_map1 - coadd_map2).max() < 1e-14 
            assert np.abs(coadd_map1 - mask*input_map).max() < 1e-14
        
            if fill is True:
                outside = (1-mask) * input_map
                coadd_map1 += outside
                coadd_map2 += outside
                
                assert np.abs(coadd_map1 - input_map).max() < 1e-14
                
                f1 = nmt.NmtField(np.ones_like(mask), [coadd_map1[1], coadd_map1[2]])
                f2 = nmt.NmtField(np.ones_like(mask), [coadd_map2[1], coadd_map2[2]])
                
            else:
                f1 = nmt.NmtField(mask_apo, [coadd_map1[1], coadd_map1[2]], purify_b=pure_b)
                f2 = nmt.NmtField(mask_apo, [coadd_map2[1], coadd_map2[2]], purify_b=pure_b)            
            
        else:
            f1 = nmt.NmtField(mask_apo, [coadd_map1[1], coadd_map1[2]], purify_b=pure_b)
            f2 = nmt.NmtField(mask_apo, [coadd_map2[1], coadd_map2[2]], purify_b=pure_b)

        Cls[i] = compute_master(f1, f2, w, leakage)

    return np.mean(Cls, axis=0), np.std(Cls, axis=0)