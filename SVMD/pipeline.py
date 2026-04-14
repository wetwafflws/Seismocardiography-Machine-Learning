import numpy as np
from scipy.signal import lfilter, hilbert, find_peaks

def apply_mti_filter(signal_in, beta):
    # The MTI filter is essentially a high-pass filter 
    # Transfer function equivalent: b = [1-beta], a = [1, -beta]
    b = [1 - beta]
    a = [1, -beta]
    s_R = lfilter(b, a, signal_in)
    return signal_in - s_R

def preprocess_scg(signal_in):
    # Bandpass filter via two high-pass MTI filters
    x_b1 = apply_mti_filter(signal_in, 0.90)
    x_b2 = apply_mti_filter(signal_in, 0.99)
    y = x_b2 - x_b1
    return signal_in 

def svmd_extract_modes(signal_in, max_modes=5, max_alpha=20000, tol=1e-6):
    """
    A streamlined Python port of the SVMD algorithm's core frequency-domain updates.
    Returns a list of extracted IMFs.
    """
    T = len(signal_in)
    t = np.arange(1, T + 1) / T
    omega_freqs = t - 0.5 - 1/T
    
    # FFT and make one-sided
    f_hat = np.fft.fftshift(np.fft.fft(signal_in))
    f_hat_onesided = f_hat.copy()
    f_hat_onesided[:T//2] = 0
    
    imfs = []
    u_hat_i = np.zeros_like(f_hat_onesided, dtype=complex)
    
    alpha = 1000 # Defaulting starting alpha
    tau = 0
    
    # Extract modes iteratively
    for k in range(max_modes):
        u_hat_k = np.zeros_like(f_hat_onesided, dtype=complex)
        omega_k = 0.0
        lambda_k = np.zeros_like(f_hat_onesided, dtype=complex)
        
        udiff = tol + 1
        n_iters = 0
        
        while udiff > tol and n_iters < 100:
            u_hat_k_old = u_hat_k.copy()
            
            # Update mode u_k
            denom = 1 + (alpha**2) * (omega_freqs - omega_k)**4
            u_hat_k = (f_hat_onesided - u_hat_i + lambda_k/2) / denom
            
            # Update center frequency omega_k
            power_spectrum = np.abs(u_hat_k[T//2:])**2
            if np.sum(power_spectrum) > 0:
                omega_k = np.sum(omega_freqs[T//2:] * power_spectrum) / np.sum(power_spectrum)
            
            # Dual ascent
            lambda_k += tau * (f_hat_onesided - (u_hat_k + u_hat_i))
            
            # Check convergence
            if np.sum(np.abs(u_hat_k_old)**2) > 0:
                udiff = np.sum(np.abs(u_hat_k - u_hat_k_old)**2) / np.sum(np.abs(u_hat_k_old)**2)
            n_iters += 1
            
        u_hat_i += u_hat_k
        
        # Reconstruct time-domain signal for this mode
        u_hat_full = np.zeros_like(u_hat_k)
        u_hat_full[T//2:] = u_hat_k[T//2:]
        u_hat_full[:T//2] = np.conj(u_hat_k[T//2:][::-1])
        
        u_k_time = np.real(np.fft.ifft(np.fft.ifftshift(u_hat_full)))
        imfs.append(u_k_time)
        
    return np.array(imfs)

def compute_waveform_factor(imf):
    # Calculates the pulsatile nature of the IMF
    rms = np.sqrt(np.mean(imf**2))
    mad = np.mean(np.abs(imf))
    if mad == 0:
        return 0
    return rms / mad

def select_and_reconstruct_ao(imfs):
    wfs = [compute_waveform_factor(imf) for imf in imfs]
    mean_wf = np.mean(wfs)
    
    # Select IMFs where WF is greater than the average
    selected_imfs = [imfs[i] for i in range(len(imfs)) if wfs[i] > mean_wf]
    
    if not selected_imfs:
        return np.zeros_like(imfs[0]), imfs, []
        
    s_AO = np.sum(selected_imfs, axis=0)
    return s_AO, imfs, [i for i, wf in enumerate(wfs) if wf > mean_wf]

def extract_ao_peaks(s_AO, fs=256):
    # Seventh power law
    s_AO_7 = s_AO**7
    
    # Envelope extraction via Hilbert transform
    envelope = np.abs(hilbert(s_AO_7))
    
    # Envelope smoothing (moving average window of ~1/10s)
    window_len = int(fs / 10)
    if window_len < 1: window_len = 1
    smoothed_env = np.convolve(envelope, np.ones(window_len)/window_len, mode='same')
    
    # Peak detection
    peaks, _ = find_peaks(smoothed_env, distance=fs*0.4) # Assume max 150 BPM -> min dist ~0.4s
    return smoothed_env, peaks