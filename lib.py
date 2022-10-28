import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft
from scipy.linalg import block_diag

def generate_map(Dl, nside, pix_size):
    ell = np.arange(len(Dl))
    Cl = Dl * 2 * np.pi / (ell*(ell+1))
    Cl[0] = 0 
    Cl[1] = 0
    
    x = np.linspace(-0.5, 0.5, nside)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    pix_to_rad = pix_size/60 * np.pi/180 
    ell_scale_factor = 2 * np.pi / pix_to_rad  
    ell2d = R * ell_scale_factor 
    Cl_expanded = np.zeros(int(ell2d.max())+1) 
    Cl_expanded[0:Cl.size] = Cl # fill in the Cls until the max of the Cl vector

    Cl2d = Cl_expanded[ell2d.astype(int)] 
    plt.imshow(np.log(Cl2d))
    plt.show()

    random_array = np.random.normal(0,1,(nside,nside))
    FT_random_array = np.fft.fft2(random_array)   
    FT_2d = np.sqrt(Cl2d) * FT_random_array 
#     plt.imshow(FT_2d.real)
#     plt.show()

    m = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
    m /= pix_to_rad
    return m.real

def generate_pointing(nside):
    exp = int(np.log(nside)/np.log(2))
    
    P_lr = block_diag(np.eye(nside), np.flip(np.eye(nside), axis=1))

    for i in range(exp-1):
        P_lr = block_diag(P_lr, P_lr)

    nsamp = P_lr.shape[0]
    P_lr = np.flip(P_lr, axis=0)

    P_ud = np.zeros((nsamp, nsamp))

    for i, row in enumerate(P_ud):
        chunk = i // nside
        index = i * nside - (chunk * nside**2) + chunk
        if chunk % 2:
            step_size = chunk*nside-(i-nside+1)
            index = nside * step_size + chunk
        row[index]=1

    P = np.vstack([P_lr, P_ud])    
    return P

def generate_noise(nsamp, dt, fknee, alpha, sigma):
    freq = np.abs(np.fft.fftfreq(nsamp, dt))
    noise_spec = (1+(np.maximum(freq,freq[1])/fknee)**-alpha)*sigma**2
    plt.loglog(freq, noise_spec)
    plt.grid()
    plt.show()
    rand = np.fft.fft(np.random.default_rng().standard_normal(nsamp))
    return np.fft.ifft(rand * noise_spec**0.5).real

def generate_mask(m, pix_size, ell_cutoff):
    nside = m.shape[0]
    x = np.linspace(-0.5, 0.5, nside)
    X, Y = np.meshgrid(x, x)
    R = np.fft.fftshift(np.sqrt(X**2 + Y**2))
    
    pix_to_rad = pix_size/60 * np.pi/180 
    ell_scale_factor = 2 * np.pi / pix_to_rad  
    ell2d = R * ell_scale_factor 
    
    ell2d[(ell2d >= ell_cutoff)] =0
    ell2d[(ell2d!=0)] = 1
    ell2d_c = 1 - ell2d
    
    dft1 = dft(nside)
    D = np.kron(dft1, dft1)
    D_inv = np.linalg.inv(D)    
    
    fft2_map = (D @ m.flatten()).reshape(nside, nside)
    fft2_map_inv = np.linalg.inv(fft2_map)
        
    M = np.outer((ell2d * fft2_map).flatten(), fft2_map_inv.flatten())
    M_c = np.outer((ell2d_c * fft2_map).flatten(), fft2_map_inv.flatten())
    
    recon_fft = M @ fft2_map.flatten() + M_c @ fft2_map.flatten()
    factor = np.nanmean(recon_fft / fft2_map.flatten())
    M /= factor
    M_c /= factor
        
    return M, M_c, D, D_inv, ell2d, ell2d_c

def generate_baselines(baseline_length, nsamp, rate):
    baseline_length_samp = rate * baseline_length
    n_baseline = nsamp // baseline_length_samp
    remainder = nsamp % baseline_length_samp

    F = np.zeros((n_baseline+1, nsamp))

    offset = 0 
    for i in range(n_baseline+1):    
        F[i][offset:offset+baseline_length_samp] = 1
        offset+=baseline_length_samp
            
    return F.T