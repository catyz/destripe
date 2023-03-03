import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
from tqdm import tqdm

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

def subscan_polyfilter(times, tod, n_sub=200, deg=10):
    subscans = np.array_split(np.copy(tod), n_sub)
    subscans_times = np.array_split(times, n_sub)
    for time, scan in zip(subscans_times, subscans):
        poly = np.polynomial.polynomial.Polynomial.fit(time, scan, deg)
        scan -= poly(time)
    return np.concatenate(subscans)

def map2map(input_map, pixels, rate, noise_params, subscan_poly_deg=10, replace=False, plot=True):
    nside = 128
    npix = 12*nside**2
    nsamp = len(pixels)    
    times = np.linspace(0, nsamp/rate, nsamp)

    signal = input_map[pixels]
    noise = np.zeros_like(signal)
    
    if noise_params is not None:
        fknee, alpha, sigma, seed = noise_params
        noise = generate_noise(nsamp, 1/rate, fknee, alpha, sigma, seed)
        
    tod = signal + noise
    filtered_tod = subscan_polyfilter(times, tod, n_sub=200, deg=subscan_poly_deg)
    
    if replace:
        m_planck = hp.smoothing(input_map, sigma=np.deg2rad(0)) 
        planck_tod = m_planck[pixels]
        filtered_planck_tod = subscan_polyfilter(times, planck_tod, n_sub=200, deg=subscan_poly_deg)
        lost_signal = planck_tod - filtered_planck_tod
        filtered_tod += lost_signal
        
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(15, 5))
        ax[0].plot(times, tod, label='unfiltered', alpha=0.75)
        ax[0].plot(times, filtered_tod, label='filtered', alpha=0.75)
        ax[0].plot(times, signal, label='signal', alpha=0.75)
                
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

def cl2cl(input_cl, pixels, rate, n_obs, noise_params, bin_size, replace=False):
    input_map = hp.synfast(input_cl, nside=128)
    npix = 12*hp.get_nside(input_map)**2 
    pixels_lr, pixels_ud = pixels
    mask = np.zeros(npix)
    mask[pixels_lr] = 1
    mask_apo = nmt.mask_apodization(mask, 2.5, apotype="Smooth")
    
    coadd_map1 = np.zeros(npix)
    coadd_map2 = np.zeros(npix)

    for i in range(n_obs):
        coadd_map1 += map2map(input_map, pixels_lr, rate, noise_params, subscan_poly_deg=10, replace=replace, plot=False)
        coadd_map1 += map2map(input_map, pixels_ud, rate, noise_params, subscan_poly_deg=10, replace=replace, plot=False)
        
        coadd_map2 += map2map(input_map, pixels_lr, rate, noise_params, subscan_poly_deg=10, replace=replace, plot=False)
        coadd_map2 += map2map(input_map, pixels_ud, rate, noise_params, subscan_poly_deg=10, replace=replace, plot=False)

    coadd_map1 /= 2*n_obs
    coadd_map2 /= 2*n_obs

    cross_cl, ells = map2cl(bin_size, mask_apo, coadd_map1, coadd_map2)
    
    return cross_cl[0], ells

def map2cl(bin_size, mask_apo, m1, m2=None):    
    nside = hp.get_nside(mask_apo)
    b = nmt.NmtBin.from_nside_linear(nside, bin_size)
    ells = b.get_effective_ells()

    f = nmt.NmtField(mask_apo, [m1])
    if m2 is None:
        cl = nmt.compute_full_master(f, f, b)
    else:
        g = nmt.NmtField(mask_apo, [m2])
        cl = nmt.compute_full_master(f, g, b)
    
    return cl, ells

def sim_cl2cl(input_cl, pixels, rate, n_sims=1, n_obs=1, noise_params=(0.5,2,100,None), bin_size=8, replace=False):
    cl = []
    for i in tqdm(range(n_sims)):
        cross_cl, ells = cl2cl(input_cl, pixels, rate, n_obs, noise_params, bin_size, replace)
        cl.append(cross_cl)
    
    return ells, np.array(cl)