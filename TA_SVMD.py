import streamlit as st
import wfdb
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import json
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.stats import zscore


def format_timestamp(seconds_val):
    """Converts seconds into HH:MM:SS.ffff format"""
    hours = int(seconds_val // 3600)
    minutes = int((seconds_val % 3600) // 60)
    seconds = seconds_val % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:07.4f}"


def save_peaks_to_json(peaks_indices, fs, record_name, output_dir="Saved_Peaks"):
    """Saves peak indices to a JSON file in the requested format"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert indices to seconds, then to formatted timestamp strings
    timestamps = [format_timestamp(p / fs) for p in peaks_indices]

    # Create dictionary matching the requested format
    data = {f"{record_name}_AO_Peaks": timestamps}

    out_path = os.path.join(output_dir, f"{record_name}_AO_Peaks.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

    return out_path

# ==========================================
# USER CONFIGURATION
# ==========================================
DEFAULT_CEBS_DB_PATH = r"C:\Users\Codbuck\Documents\Sem7_Sync\Prata\SVMD\Data\CEBS_Data\files"
DEFAULT_UNHEALTHY_BASE_DIR = r"Data"
# ==========================================

st.set_page_config(page_title="CEBS Data Viewer & Replication", layout="wide")

# ==========================================
# 1. ALGORITHM IMPLEMENTATION (SVMD)
# ==========================================

def svmd(signal_in, max_alpha=2000, tau=0, tol=1e-6, stopc=1, init_omega=0):
    """
    Successive Variational Mode Decomposition (SVMD)
    A drop-in Python replacement corresponding to the original MATLAB script.
    """
    signal_in = np.array(signal_in).flatten()
    if len(signal_in) % 2 > 0:
        signal_in = signal_in[:-1]
        
    # Estimate the noise via Savitzky-Golay filter (matches MATLAB: sgolayfilt(signal,8,25))
    y = savgol_filter(signal_in, window_length=25, polyorder=8)
    signoise = signal_in - y
    
    save_T = len(signal_in)
    fs = 1.0 / save_T
    
    # Mirror the signal and noise to extend (Part 1)
    T_half = save_T // 2
    f_mir = np.zeros(2 * save_T)
    f_mir_noise = np.zeros(2 * save_T)
    
    f_mir[0:T_half] = signal_in[T_half-1::-1]
    f_mir_noise[0:T_half] = signoise[T_half-1::-1]
    
    f_mir[T_half:save_T + T_half] = signal_in
    f_mir_noise[T_half:save_T + T_half] = signoise
    
    f_mir[save_T + T_half:2 * save_T] = signal_in[save_T-1:T_half-1:-1]
    f_mir_noise[save_T + T_half:2 * save_T] = signoise[save_T-1:T_half-1:-1]
    
    f = f_mir
    fnoise = f_mir_noise
    T = len(f)
    
    t = np.arange(1, T + 1) / T
    omega_freqs = t - 0.5 - 1.0/T
    
    # FFT and one-sided preparations
    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_onesided = f_hat.copy()
    f_hat_onesided[0:T//2] = 0
    
    f_hat_n = np.fft.fftshift(np.fft.fft(fnoise))
    f_hat_n_onesided = f_hat_n.copy()
    f_hat_n_onesided[0:T//2] = 0
    
    noisepe = np.linalg.norm(f_hat_n_onesided, 2)**2
    
    N_max = 300
    omega_L = np.zeros(N_max)
    if init_omega == 0:
        omega_L[0] = 0.0
    else:
        omega_L[0] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1)))[0]
        
    minAlpha = 10
    Alpha = minAlpha
    
    n = 0
    m_val = 0
    SC2 = 0
    l = 0
    bf = 0
    
    BIC = []
    h_hat_Temp = []
    u_hat_Temp = []
    u_hat_i = []
    alpha_arr = []
    
    n2 = 0
    polm = []
    polm_temp = 1.0
    
    omega_d_Temp = []
    sigerror = []
    gamma = []
    normind = []
    
    lambda_arr = np.zeros((N_max, len(omega_freqs)), dtype=complex)
    u_hat_L = np.zeros((N_max, len(omega_freqs)), dtype=complex)
    
    # Part 2: Main loop for iterative updates
    while SC2 != 1:
        while Alpha < (max_alpha + 1):
            udiff = tol + np.finfo(float).eps
            while udiff > tol and n < N_max - 1:
                
                sum_h_hat = np.sum(h_hat_Temp, axis=0) if len(h_hat_Temp) > 0 else np.zeros(len(omega_freqs))
                sum_u_i = np.sum(u_hat_i, axis=0) if len(u_hat_i) > 0 else np.zeros(len(omega_freqs), dtype=complex)
                
                freq_diff = omega_freqs - omega_L[n]
                freq_diff_sq = freq_diff**2
                freq_diff_pow4 = freq_diff_sq**2
                alpha_sq = Alpha**2
                
                # Update u_hat_L
                num_u = f_hat_onesided + (alpha_sq * freq_diff_pow4) * u_hat_L[n, :] + lambda_arr[n, :] / 2.0
                den_u = 1 + alpha_sq * freq_diff_pow4 * (1 + 2 * Alpha * freq_diff_sq) + sum_h_hat
                u_hat_L[n+1, :] = num_u / den_u
                
                # Update omega_L
                pos = slice(T//2, T)
                mag_sq = np.abs(u_hat_L[n+1, pos])**2
                den_omega = np.sum(mag_sq)
                if den_omega > 0:
                    omega_L[n+1] = np.dot(omega_freqs[pos], mag_sq) / den_omega
                else:
                    omega_L[n+1] = omega_L[n]
                    
                # Update lambda (dual ascent)
                term1 = alpha_sq * freq_diff_pow4
                inner_paren = f_hat_onesided - u_hat_L[n+1, :] - sum_u_i + lambda_arr[n, :] / 2.0
                num_tau = term1 * inner_paren - sum_u_i
                den_tau = 1 + term1
                
                lambda_arr[n+1, :] = lambda_arr[n, :] + tau * (f_hat_onesided - (u_hat_L[n+1, :] + num_tau / den_tau) + sum_u_i)
                
                # Update criteria
                udiff = np.finfo(float).eps
                diff_u = u_hat_L[n+1, :] - u_hat_L[n, :]
                num_udiff = np.sum(diff_u * np.conj(diff_u)).real / T
                den_udiff = np.sum(u_hat_L[n, :] * np.conj(u_hat_L[n, :])).real / T
                
                if den_udiff > 0:
                    udiff += np.abs(num_udiff / den_udiff)
                udiff = np.abs(udiff)
                
                n += 1
                
            # Part 3: Increasing Alpha
            if abs(m_val - np.log(max_alpha)) > 1:
                m_val += 1
            else:
                m_val += 0.05
                bf += 1
                
            if bf >= 2:
                Alpha += 1
                
            if Alpha <= (max_alpha - 1):
                if bf == 1:
                    Alpha = max_alpha - 1
                else:
                    Alpha = np.exp(m_val)
                
                omega_L[0] = omega_L[n]
                temp_ud = u_hat_L[n, :].copy()
                n = 0
                lambda_arr = np.zeros((N_max, len(omega_freqs)), dtype=complex)
                u_hat_L = np.zeros((N_max, len(omega_freqs)), dtype=complex)
                u_hat_L[0, :] = temp_ud

        # Part 4: Saving the Modes and Center Frequencies
        valid_omegas = omega_L[omega_L > 0]
        if len(valid_omegas) > 0:
            idx = max(0, min(n - 1, len(valid_omegas) - 1))
            omega_d_Temp.append(valid_omegas[idx])
        else:
            omega_d_Temp.append(0.0)
            
        u_hat_Temp.append(u_hat_L[n, :].copy())
        alpha_arr.append(Alpha)
        
        Alpha = minAlpha
        bf = 0
        
        # Initialize omega_L for the next run
        if init_omega > 0:
            ii = 0
            while ii < 1 and n2 < 300:
                rand_omega = np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand())
                checkp = np.abs(np.array(omega_d_Temp) - rand_omega)
                if len(checkp[checkp < 0.02]) <= 0:
                    ii = 1
                n2 += 1
            omega_L_start = rand_omega
        else:
            omega_L_start = 0.0
            
        lambda_arr = np.zeros((N_max, len(omega_freqs)), dtype=complex)
        gamma.append(1)
        
        h_hat_new = gamma[l] / ((alpha_arr[l]**2) * (omega_freqs - omega_d_Temp[l])**4)
        h_hat_Temp.append(h_hat_new)
        u_hat_i.append(u_hat_Temp[l])
        
        sum_u_i = np.sum(u_hat_i, axis=0) if len(u_hat_i) > 0 else np.zeros(len(omega_freqs), dtype=complex)
        
        # Part 5: Stopping Criteria Handling
        if stopc == 1:
            # Presence of Noise
            sigerror_val = np.linalg.norm(f_hat_onesided - sum_u_i, 2)**2
            sigerror.append(sigerror_val)
            if n2 >= 300 or sigerror[l] <= np.round(noisepe):
                SC2 = 1
                
        elif stopc == 2:
            # Exact Reconstruction
            sum_u_temp = np.sum(u_hat_Temp, axis=0)
            normind_val = (1/T) * np.linalg.norm(sum_u_temp - f_hat_onesided, 2)**2 / ((1/T) * np.linalg.norm(f_hat_onesided, 2)**2)
            normind.append(normind_val)
            if n2 >= 300 or normind[l] < 0.005:
                SC2 = 1
                
        elif stopc == 3:
            # Bayesian Method
            sigerror_val = np.linalg.norm(f_hat_onesided - sum_u_i, 2)**2
            sigerror.append(sigerror_val)
            BIC_val = 2 * T * np.log(sigerror[l]) + (3 * (l + 1)) * np.log(2 * T)
            BIC.append(BIC_val)
            if l > 0 and BIC[l] > BIC[l-1]:
                SC2 = 1
                
        else:
            # Power of the Last Mode
            term = (4 * alpha_arr[l] * u_hat_i[l]) / (1 + 2 * alpha_arr[l] * (omega_freqs - omega_d_Temp[l])**2)
            dot_prod = np.sum(term * np.conj(u_hat_i[l]))
            polm_val = np.abs(dot_prod)
            
            if l == 0:
                polm.append(polm_val)
                polm_temp = polm[0]
                polm[0] = 1.0
            else:
                polm.append(polm_val / polm_temp)
                
            if l > 0 and abs(polm[l] - polm[l-1]) < 0.001:
                SC2 = 1
                
        # Part 6: Reset the counters
        u_hat_L = np.zeros((N_max, len(omega_freqs)), dtype=complex)
        omega_L = np.zeros(N_max)
        omega_L[0] = omega_L_start
        n = 0
        l += 1
        m_val = 0
        n2 = 0
        
    # Part 7: Signal Reconstruction
    omega_final = np.array(omega_d_Temp)
    L = len(omega_final)
    u_hat = np.zeros((T, L), dtype=complex)
    
    for idx in range(L):
        u_hat[T//2:T, idx] = u_hat_Temp[idx][T//2:T]
        u_hat[T//2:0:-1, idx] = np.conj(u_hat_Temp[idx][T//2:T])
        u_hat[0, idx] = np.conj(u_hat[-1, idx])
        
    u = np.zeros((L, T))
    for idx in range(L):
        u[idx, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, idx])))
        
    indic = np.argsort(omega_final)
    omega_final = omega_final[indic]
    u = u[indic, :]
    
    # Remove mirror part to match original size
    u_final = u[:, T//4 : 3*T//4]
    
    return u_final, omega_final

# ==========================================
# 2. PREPROCESSING 
# ==========================================

def load_record_names(path):
    if not os.path.exists(path): return []
    files = [f.replace('.hea', '') for f in os.listdir(path) if f.endswith('.hea')]
    files.sort()
    return files

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def get_unhealthy_patient_ids():
    cp_ids = [f"CP-{i:02d}" for i in range(1, 71)]
    up_ids = [f"UP-{i:02d}" for i in range(1, 31)]
    return cp_ids + up_ids

@st.cache_data(show_spinner=False)
def load_unhealthy_labels(base_dir):
    labels_file = os.path.join(base_dir, "ground_truth_labels.csv")
    if not os.path.exists(labels_file):
        return pd.DataFrame()
    labels_df = pd.read_csv(labels_file)
    labels_df.set_index("Patient ID", inplace=True)
    return labels_df

def get_unhealthy_label_str(labels_df, patient_id):
    if labels_df.empty:
        return "Unknown (labels file not found)"
    try:
        patient_conditions = labels_df.loc[patient_id]
        active_conditions = patient_conditions[patient_conditions > 0]
        if len(active_conditions) == 0:
            return "Normal"
        condition_names = active_conditions.index.tolist()
        cleaned_names = [name.replace("Moderate or greater ", "") for name in condition_names]
        return ", ".join(cleaned_names)
    except KeyError:
        return f"Unknown (Patient ID {patient_id} not in labels file)"

@st.cache_data(show_spinner=False)
def load_unhealthy_patient_data(patient_id, base_dir, target_fs=256):
    raw_file = os.path.join(base_dir, f"Cleaned_{patient_id}.csv")
    json_file = os.path.join(base_dir, f"{patient_id}-ECG.json")

    if not os.path.exists(raw_file) or not os.path.exists(json_file):
        raise FileNotFoundError(
            f"Missing data for {patient_id}. Expected {raw_file} and {json_file}."
        )

    with open(json_file, "r") as f:
        ecg_annotations = json.load(f)

    r_peaks_timestamps = next(iter(ecg_annotations.values()))
    r_peaks_seconds = pd.to_timedelta(r_peaks_timestamps).total_seconds().to_numpy()
    r_peak_indices = (r_peaks_seconds * target_fs).astype(int)

    raw_data = pd.read_csv(raw_file)
    col_map = {
        "AccX": "Accel_X",
        "AccY": "Accel_Y",
        "AccZ": "Accel_Z",
        "ECG": "ECG",
    }

    if not all(col in raw_data.columns for col in col_map.keys()):
        raise ValueError(f"Required columns missing in {raw_file}.")

    df_clean = raw_data.rename(columns=col_map)

    prefix, num_str = patient_id.split("-")
    patient_num = int(num_str)
    if prefix == "UP" and patient_num >= 22:
        original_len = len(df_clean)
        new_len = int(original_len * target_fs / 512)
        resampled_data = {}
        for col in ["Accel_X", "Accel_Y", "Accel_Z", "ECG"]:
            resampled_data[col] = signal.resample(df_clean[col], new_len)
        df_full = pd.DataFrame(resampled_data)
    else:
        df_full = df_clean.copy()

    df_full["scg_z_filt"] = butter_bandpass_filter(df_full["Accel_Z"], 1.0, 30.0, target_fs)
    df_full["scg_z_norm"] = zscore(df_full["scg_z_filt"])
    df_full["ecg_filt"] = butter_bandpass_filter(df_full["ECG"], 1.0, 40.0, target_fs)

    signals = np.zeros((len(df_full), 4))
    signals[:, 0] = df_full["ecg_filt"].to_numpy()
    signals[:, 3] = df_full["scg_z_norm"].to_numpy()

    return signals, r_peak_indices, target_fs

def apply_mti_filter(raw_signal, fs_original):
    target_fs = 500
    if fs_original > target_fs:
        num_samples = int(len(raw_signal) * (target_fs / fs_original))
        s_downsampled = signal.resample(raw_signal, num_samples)
    else:
        s_downsampled = raw_signal
        
    def mti_pass(sig, beta):
        return signal.lfilter([beta, -beta], [1, -beta], sig)

    x_beta1 = mti_pass(s_downsampled, 0.9)
    x_beta2 = mti_pass(s_downsampled, 0.99)
    y_filtered = x_beta2 - x_beta1
    
    if fs_original > target_fs:
        y_final = signal.resample(y_filtered, len(raw_signal))
    else:
        y_final = y_filtered
        
    y_detrended = signal.detrend(y_final)
    y_smoothed = signal.medfilt(y_detrended, kernel_size=5)
    
    return y_smoothed

# ==========================================
# 3. WAVEFORM FACTOR & RECONSTRUCTION
# ==========================================

def select_ao_modes(modes, omegas=None, fs=None, freq_cutoff_hz=50):
    wfs = []
    for mode in modes:
        rms = np.sqrt(np.mean(mode**2))
        mav = np.mean(np.abs(mode))
        wfs.append(rms / mav)
        
    wfs = np.array(wfs)
    
    if omegas is not None and fs is not None:
        center_freq_hz = np.abs(omegas) * fs
        eligible_mask = center_freq_hz <= freq_cutoff_hz
    else:
        eligible_mask = np.ones(len(wfs), dtype=bool)
    
    if np.any(eligible_mask):
        wf_mean = np.mean(wfs[eligible_mask])
        selected_indices = np.where((wfs > wf_mean) & eligible_mask)[0]
    else:
        wf_mean = np.nan
        selected_indices = np.array([], dtype=int)
    
    if len(selected_indices) > 0:
        s_ao = np.sum(modes[selected_indices], axis=0)
    else:
        s_ao = np.zeros_like(modes[0])
        
    return s_ao, wfs, wf_mean, selected_indices

# ==========================================
# 4. ENVELOPE & PEAK DETECTION
# ==========================================

def extract_ao_peaks(s_ao, fs, prominence_factor=0.05, power=7):
    # NEW: Apply a Tukey window to smoothly taper the edges to zero
    # alpha=0.05 means only the first and last 2.5% of the window are faded out
    taper_window = signal.windows.tukey(len(s_ao), alpha=0.05) 
    s_ao_tapered = s_ao * taper_window
    
    # 1. Power Law Detection (using the tapered signal)
    s_ao_7 = np.power(s_ao_tapered, power)
    
    # 2. Envelope extraction via Hilbert Transform
    analytic_signal = signal.hilbert(s_ao_7)
    envelope = np.abs(analytic_signal)
    
    # 3. Envelope smoothing (Sliding Average Filter, T = 1/10s)
    window_width = int(fs / 10) 
    smoothed_env = np.convolve(envelope, np.ones(window_width)/window_width, mode='same')
    
    # 4. Detect Peaks
    min_distance = int(0.4 * fs)
    # Dynamic prominence threshold
    peaks, _ = find_peaks(smoothed_env, distance=min_distance, prominence=np.max(smoothed_env)*prominence_factor)
    
    return s_ao_7, envelope, smoothed_env, peaks

def detect_r_peaks(ecg_signal, fs):
    """
    Detect R peaks in ECG signal using Pan-Tompkins-inspired algorithm.
    """
    # Apply Tukey window to taper edges (same as SCG processing)
    taper_window = signal.windows.tukey(len(ecg_signal), alpha=0.05)
    ecg_tapered = ecg_signal * taper_window
    
    # Downsample to 500 Hz if fs is too high for numerical stability
    if fs > 1000:
        target_fs = 500
        downsample_factor = int(fs / target_fs)
        ecg_downsampled = signal.decimate(ecg_tapered, downsample_factor, zero_phase=True)
        fs_working = fs / downsample_factor
    else:
        ecg_downsampled = ecg_tapered
        fs_working = fs
    
    # 1. Bandpass filter (5-15 Hz for QRS detection)
    nyquist = fs_working / 2
    low = 5.0 / nyquist
    high = 15.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    ecg_filtered = signal.filtfilt(b, a, ecg_downsampled)
    
    # 2. Derivative (emphasizes QRS slope)
    ecg_diff = np.diff(ecg_filtered)
    ecg_diff = np.append(ecg_diff, ecg_diff[-1])  # Maintain length
    
    # 3. Square the signal
    ecg_squared = ecg_diff ** 2
    
    # 4. Moving window integration (150ms window)
    window_size = int(0.15 * fs_working)
    ecg_integrated = np.convolve(ecg_squared, np.ones(window_size)/window_size, mode='same')
    
    # 5. Detect peaks with adaptive threshold
    threshold = 0.35 * np.max(ecg_integrated)
    min_distance = int(0.4 * fs_working)  # Minimum 0.4s between R peaks
    
    r_peaks, _ = find_peaks(ecg_integrated, height=threshold, distance=min_distance)
    
    # Scale peak indices back to original sampling rate if downsampled
    if fs > 1000:
        r_peaks = r_peaks * downsample_factor
        # Also upsample filtered signal for visualization
        ecg_filtered = signal.resample(ecg_filtered, len(ecg_signal))
        ecg_integrated = signal.resample(ecg_integrated, len(ecg_signal))
    
    return r_peaks, ecg_filtered, ecg_integrated

def compare_intervals(ao_peaks, r_peaks, fs):
    """
    Compare AO-AO intervals with R-R intervals and compute statistics.
    """
    if len(ao_peaks) < 2 or len(r_peaks) < 2:
        return None
    
    # Calculate intervals in milliseconds
    ao_intervals_ms = np.diff(ao_peaks) / fs * 1000
    rr_intervals_ms = np.diff(r_peaks) / fs * 1000
    
    # Calculate heart rates
    ao_hr = 60000 / ao_intervals_ms
    rr_hr = 60000 / rr_intervals_ms
    
    # Match intervals (use minimum length)
    min_len = min(len(ao_intervals_ms), len(rr_intervals_ms))
    ao_intervals_ms = ao_intervals_ms[:min_len]
    rr_intervals_ms = rr_intervals_ms[:min_len]
    ao_hr = ao_hr[:min_len]
    rr_hr = rr_hr[:min_len]
    
    # Compute statistics
    correlation = np.corrcoef(ao_intervals_ms, rr_intervals_ms)[0, 1]
    rmse = np.sqrt(np.mean((ao_intervals_ms - rr_intervals_ms)**2))
    mae = np.mean(np.abs(ao_intervals_ms - rr_intervals_ms))
    
    # Bland-Altman statistics
    mean_intervals = (ao_intervals_ms + rr_intervals_ms) / 2
    diff_intervals = ao_intervals_ms - rr_intervals_ms
    mean_diff = np.mean(diff_intervals)
    std_diff = np.std(diff_intervals)
    
    return {
        'ao_intervals_ms': ao_intervals_ms,
        'rr_intervals_ms': rr_intervals_ms,
        'ao_hr': ao_hr,
        'rr_hr': rr_hr,
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'mean_intervals': mean_intervals,
        'diff_intervals': diff_intervals
    }

# ==========================================
# MAIN APP UI
# ==========================================

st.title("SVMD Data Viewer & Replication")
st.markdown("Replicating **Zheng et al. (2024)**: Preprocessing, SVMD, and AO Peak Extraction.")

dataset_label = st.radio(
    "Dataset",
    ["CEBS (WFDB)", "Unhealthy (CSV+JSON)"],
    horizontal=True
)
is_unhealthy = dataset_label.startswith("Unhealthy")

db_path = None
base_dir = None

path_col, _ = st.columns([3, 1])
with path_col:
    if is_unhealthy:
        base_dir = st.text_input("Dataset Base Folder", value=DEFAULT_UNHEALTHY_BASE_DIR)
        records = get_unhealthy_patient_ids()
    else:
        db_path = st.text_input("Database Folder Path", value=DEFAULT_CEBS_DB_PATH)
        records = load_record_names(db_path)

if is_unhealthy and base_dir and not os.path.exists(base_dir):
    st.error(f"Dataset base folder not found: {base_dir}")
    st.stop()

if not records:
    st.error("No records found. Please check the dataset path.")
else:
    with st.sidebar:
        st.header("Controls")
        record_label = "Select Patient" if is_unhealthy else "Select Record"
        selected_record = st.selectbox(record_label, records)
        
        st.divider()
        st.subheader("1. Preprocessing")
        use_mti = st.checkbox(
            "Apply Complete Preprocessing (MTI, Detrend, Median)",
            value=True
        )
        
        st.divider()
        st.subheader("2. Run Algorithm")
        run_svmd_btn = st.button("Extract AO Peaks & Heart Rate")
        svmd_alpha = st.slider("SVMD Alpha (Bandwidth)", 100, 2000, 260)
        
        st.divider()
        st.subheader("3. Peak Detection")
        prominence_factor = st.slider("Peak Prominence Threshold", 0.01, 0.30, 0.05, 0.01,
                                     help="Higher values detect only the most prominent peaks (0.05 = 5% of max)")
        power_exp = st.slider(
            "Power-Law Exponent",
            3,
            9,
            7,
            help="Controls the power applied before envelope extraction."
        )

        st.divider()
        st.subheader("4. Batch (Random Segment)")
        batch_random_btn = st.button(
            "Run Random 10s Batch (Start >= 60s)",
            disabled=is_unhealthy
        )
        
        st.divider()
        st.subheader("5. Full Record Analysis")
        full_record_btn = st.button("🔍 Analyze Full Record", type="primary")
        remove_outliers = st.checkbox("Remove Outlier Intervals (IQR method)", value=False,
                                     help="Remove interval pairs where either value is beyond 1.5×IQR from Q1/Q3. Maintains equal AO-AO and R-R counts.")
        
        st.divider()
        st.subheader("6. Batch Full Record Analysis")
        unhealthy_skip_seconds = None
        if is_unhealthy:
            unhealthy_skip_seconds = st.number_input(
                "Skip first N seconds (Batch)",
                min_value=0.0,
                value=60.0,
                step=1.0,
                help="Starts batch processing after this time for every patient."
            )

        batch_full_record_btn = st.button(
            "🔄 Analyze Full Records for All (Batch)",
            type="secondary"
        )

        st.divider()
        st.subheader("7. Output Settings")
        save_json_output = st.checkbox(
            "Save AO Peaks to JSON",
            value=True,
            help="Automatically saves detected peaks to a local folder during Full Record and Batch analyses."
        )
        output_folder = st.text_input("Output Folder", value="Saved_Peaks")

        try:
            if is_unhealthy:
                signals, r_peaks_indices, fs = load_unhealthy_patient_data(selected_record, base_dir)
                total_duration = len(signals) / fs
                labels_df = load_unhealthy_labels(base_dir)
                label_str = get_unhealthy_label_str(labels_df, selected_record)
                st.success(f"Loaded {selected_record}")
                st.info(f"Patient condition: {label_str}")
            else:
                header = wfdb.rdheader(os.path.join(db_path, selected_record))
                fs = header.fs
                total_duration = header.sig_len / fs
                st.success(f"Loaded {selected_record}")

            window_size = st.slider("Window Size (s)", 1, 30, 10)
            start_time = st.slider("Start Time (s)", 0.0, total_duration - window_size, 0.0, 0.1)
        except Exception as e:
            st.error(f"Error reading record: {e}")
            st.stop()

    try:
        if is_unhealthy:
            signals, r_peaks_indices, fs = load_unhealthy_patient_data(selected_record, base_dir)
            labels_df = load_unhealthy_labels(base_dir)
            label_str = get_unhealthy_label_str(labels_df, selected_record)
            st.subheader(f"Patient Condition: {label_str}")
        else:
            record = wfdb.rdsamp(os.path.join(db_path, selected_record))
            signals = record[0]
        
        anns = None
        try:
            annotation = wfdb.rdann(os.path.join(db_path, selected_record), 'atr')
            anns = annotation.sample
        except: pass

        # Basic Plot Generation
        scg_filtered_full = apply_mti_filter(signals[:, 3], fs) if use_mti else signals[:, 3]
        
        start_idx = int(start_time * fs)
        end_idx = int((start_time + window_size) * fs)
        time_axis = np.linspace(start_time, start_time + window_size, end_idx - start_idx)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Raw ECG I", "SCG Signal (Original vs Preprocessed)"])
        fig.add_trace(go.Scatter(x=time_axis, y=signals[start_idx:end_idx, 0], mode='lines', line=dict(color='black', width=1), name="ECG I"), row=1, col=1)
        
        if use_mti:
            fig.add_trace(go.Scatter(x=time_axis, y=signals[start_idx:end_idx, 3], mode='lines', line=dict(color='gray', width=1), opacity=0.5, name="Raw SCG"), row=2, col=1)
            fig.add_trace(go.Scatter(x=time_axis, y=scg_filtered_full[start_idx:end_idx], mode='lines', line=dict(color='navy', width=1.5), name="Fully Filtered SCG"), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=time_axis, y=signals[start_idx:end_idx, 3], mode='lines', line=dict(color='black', width=1), name="Raw SCG"), row=2, col=1)
            
        fig.update_layout(height=500, margin=dict(t=40, b=40), hovermode='x unified', plot_bgcolor='white')
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 0, 0, 0.2)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255, 0, 0, 0.2)')
        st.plotly_chart(fig, use_container_width=True)
        
        # PSD Comparison Plot
        st.markdown("### 📡 Frequency Spectrum Analysis")
        st.write("Comparison of Power Spectral Density (PSD) before and after preprocessing")
        
        # Calculate PSD using Welch's method
        scg_raw_segment = signals[start_idx:end_idx, 3]
        scg_preproc_segment = scg_filtered_full[start_idx:end_idx]
        
        f_raw, psd_raw = signal.welch(scg_raw_segment, fs=fs, nperseg=min(256, len(scg_raw_segment)//2))
        f_preproc, psd_preproc = signal.welch(scg_preproc_segment, fs=fs, nperseg=min(256, len(scg_preproc_segment)//2))
        
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(
            x=f_raw, y=10*np.log10(psd_raw + 1e-10), 
            mode='lines', 
            name='Raw SCG',
            line=dict(color='gray', width=1.5)
        ))
        fig_psd.add_trace(go.Scatter(
            x=f_preproc, y=10*np.log10(psd_preproc + 1e-10), 
            mode='lines', 
            name='Preprocessed SCG',
            line=dict(color='navy', width=2)
        ))
        
        fig_psd.update_layout(
            title="Power Spectral Density (Welch's Method)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            margin=dict(t=60, b=40)
        )
        fig_psd.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
        fig_psd.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
        
        st.plotly_chart(fig_psd, use_container_width=True)
        
        # Algorithm Execution
        if run_svmd_btn:
            if not use_mti and not is_unhealthy:
                st.warning("Please enable 'Apply Complete Preprocessing' first.")
            else:
                scg_segment = scg_filtered_full[start_idx:end_idx]
                target_fs_svmd = 500
                q = fs / target_fs_svmd
                if q > 1:
                    scg_for_svmd = signal.resample(scg_segment, int(len(scg_segment) / q))
                    fs_svmd = target_fs_svmd
                else:
                    scg_for_svmd = scg_segment
                    fs_svmd = fs

                with st.spinner("Running SVMD..."):
                    modes, omegas = svmd(scg_for_svmd, max_alpha=svmd_alpha, tau=0, stopc=3)

                if len(omegas) > 0:
                    if q > 1:
                        time_axis_imf = np.linspace(start_time, start_time + window_size, len(modes[0]))
                    else:
                        time_axis_imf = time_axis

                    st.session_state["svmd_result"] = {
                        "modes": modes,
                        "omegas": omegas,
                        "q": q,
                        "fs_svmd": fs_svmd,
                        "scg_segment": scg_segment,
                        "time_axis_imf": time_axis_imf,
                        "time_axis": time_axis,
                        "fs": fs,
                    }
                    st.session_state.pop("recon_result", None)
                    st.session_state.pop("peaks_result", None)
                else:
                    st.warning("SVMD returned no modes. Try adjusting parameters.")

        if "svmd_result" in st.session_state:
            st.divider()
            st.subheader("Algorithm Results")

            svmd_result = st.session_state["svmd_result"]
            modes = svmd_result["modes"]
            omegas = svmd_result["omegas"]
            q = svmd_result["q"]
            fs_svmd = svmd_result["fs_svmd"]
            scg_segment = svmd_result["scg_segment"]
            time_axis_imf = svmd_result["time_axis_imf"]
            center_freq_hz = np.abs(omegas) * fs_svmd

            st.markdown("### Decomposed IMFs from SVMD")
            st.write(f"**Number of IMFs extracted:** {len(modes)}")

            fig_imfs = make_subplots(rows=len(modes), cols=1, shared_xaxes=True,
                                    vertical_spacing=0.02,
                                    subplot_titles=[f"IMF {i+1} (ω = {omegas[i]:.4f})" for i in range(len(modes))])

            for i, mode in enumerate(modes):
                fig_imfs.add_trace(
                    go.Scatter(x=time_axis_imf, y=mode, mode='lines',
                              line=dict(width=1), name=f"IMF {i+1}"),
                    row=i+1, col=1
                )

            fig_imfs.update_layout(
                height=200*len(modes),
                hovermode='x unified',
                plot_bgcolor='white',
                showlegend=False,
                margin=dict(t=40, b=40)
            )
            fig_imfs.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)',
                                 title_text="Time (s)", row=len(modes), col=1)
            fig_imfs.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)')

            st.plotly_chart(fig_imfs, use_container_width=True)

            st.markdown("### AO Reconstruction")
            s_ao_svmd, wfs, wf_mean, auto_selected_idx = select_ao_modes(modes, omegas, fs_svmd)

            manual_select = st.checkbox("Manually select IMFs for reconstruction", value=False)
            selected_idx = auto_selected_idx
            if manual_select:
                selected_idx = st.multiselect(
                    "Select IMFs",
                    options=list(range(len(modes))),
                    default=list(auto_selected_idx),
                    format_func=lambda i: f"IMF {i+1} (f={center_freq_hz[i]:.2f} Hz, WF={wfs[i]:.4f})"
                )
                selected_idx = np.array(sorted(selected_idx), dtype=int)

            reconstruct_btn = st.button("Reconstruct AO")
            if reconstruct_btn:
                if len(selected_idx) > 0:
                    s_ao_svmd = np.sum(modes[selected_idx], axis=0)
                else:
                    s_ao_svmd = np.zeros_like(modes[0])

                if q > 1:
                    s_ao_plot = signal.resample(s_ao_svmd, len(scg_segment))
                else:
                    s_ao_plot = s_ao_svmd

                st.session_state["recon_result"] = {
                    "s_ao_svmd": s_ao_svmd,
                    "s_ao_plot": s_ao_plot,
                    "wfs": wfs,
                    "wf_mean": wf_mean,
                    "selected_idx": selected_idx,
                    "center_freq_hz": center_freq_hz,
                }
                st.session_state.pop("peaks_result", None)

        if "recon_result" in st.session_state:
            recon_result = st.session_state["recon_result"]
            wfs = recon_result["wfs"]
            wf_mean = recon_result["wf_mean"]
            selected_idx = recon_result["selected_idx"]
            center_freq_hz = recon_result["center_freq_hz"]

            st.markdown("### 📊 Waveform Factor Analysis")
            st.write(f"**Waveform Factor Mean:** {wf_mean:.4f}")
            st.write(f"**IMFs selected for reconstruction:** {len(selected_idx)} out of {len(wfs)}")

            imf_labels = [f"IMF {i+1}" for i in range(len(wfs))]
            colors = ['#FF4B4B' if i in selected_idx else '#D3D3D3' for i in range(len(wfs))]

            fig_wf = go.Figure()
            fig_wf.add_trace(go.Bar(
                x=imf_labels,
                y=wfs,
                marker=dict(color=colors, line=dict(color='black', width=1)),
                name='Waveform Factor',
                text=[f'{wf:.4f}' for wf in wfs],
                textposition='outside'
            ))

            fig_wf.add_hline(
                y=wf_mean,
                line_dash="dash",
                line_color="blue",
                line_width=2,
                annotation_text=f"Mean = {wf_mean:.4f}",
                annotation_position="right"
            )

            fig_wf.update_layout(
                title="Waveform Factors of IMFs (Red = Selected for Reconstruction)",
                xaxis_title="IMF Number",
                yaxis_title="Waveform Factor (RMS/MAV)",
                height=400,
                plot_bgcolor='white',
                showlegend=False,
                margin=dict(t=60, b=40)
            )
            fig_wf.update_xaxes(showgrid=False)
            fig_wf.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')

            st.plotly_chart(fig_wf, use_container_width=True)

            st.markdown("### 🎯 Center Frequency vs Waveform Factor")
            scatter_labels = [f"IMF {i+1}" for i in range(len(wfs))]
            fig_scatter = go.Figure()

            unselected = [i for i in range(len(wfs)) if i not in selected_idx]
            if unselected:
                fig_scatter.add_trace(go.Scatter(
                    x=[center_freq_hz[i] for i in unselected],
                    y=[wfs[i] for i in unselected],
                    mode='markers',
                    marker=dict(size=10, color='#D3D3D3', line=dict(color='black', width=1)),
                    text=[f"{scatter_labels[i]}<br>f={center_freq_hz[i]:.2f} Hz<br>WF={wfs[i]:.4f}" for i in unselected],
                    hovertemplate='%{text}<extra></extra>',
                    name='Not Selected',
                    showlegend=True
                ))

            if len(selected_idx) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=[center_freq_hz[i] for i in selected_idx],
                    y=[wfs[i] for i in selected_idx],
                    mode='markers',
                    marker=dict(size=12, color='#FF4B4B', line=dict(color='darkred', width=2)),
                    text=[f"{scatter_labels[i]}<br>f={center_freq_hz[i]:.2f} Hz<br>WF={wfs[i]:.4f}" for i in selected_idx],
                    hovertemplate='%{text}<extra></extra>',
                    name='Selected for AO',
                    showlegend=True
                ))

            fig_scatter.add_hline(
                y=wf_mean,
                line_dash="dash",
                line_color="blue",
                line_width=2,
                annotation_text=f"WF Threshold = {wf_mean:.4f}",
                annotation_position="right"
            )

            fig_scatter.update_layout(
                title="IMF Characteristics: Center Frequency vs Waveform Factor",
                xaxis_title="Center Frequency (Hz)",
                yaxis_title="Waveform Factor (RMS/MAV)",
                height=500,
                hovermode='closest',
                plot_bgcolor='white',
                margin=dict(t=60, b=40)
            )
            fig_scatter.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
            fig_scatter.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')

            st.plotly_chart(fig_scatter, use_container_width=True)

            detect_peaks_btn = st.button("Detect AO Peaks")
            if detect_peaks_btn:
                s_ao_plot = recon_result["s_ao_plot"]
                s_ao_7, envelope, smoothed_env, peaks = extract_ao_peaks(
                    s_ao_plot, fs, prominence_factor, power=power_exp
                )
                
                # Detect R peaks from ECG
                ecg_segment = signals[start_idx:end_idx, 0]
                r_peaks, ecg_filtered, ecg_integrated = detect_r_peaks(ecg_segment, fs)
                
                # Compare intervals
                comparison = compare_intervals(peaks, r_peaks, fs)

                st.session_state["peaks_result"] = {
                    "s_ao_7": s_ao_7,
                    "envelope": envelope,
                    "smoothed_env": smoothed_env,
                    "peaks": peaks,
                    "s_ao_plot": s_ao_plot,
                    "r_peaks": r_peaks,
                    "ecg_filtered": ecg_filtered,
                    "ecg_integrated": ecg_integrated,
                    "comparison": comparison,
                }

        if "peaks_result" in st.session_state:
            peaks_result = st.session_state["peaks_result"]
            peaks = peaks_result["peaks"]
            r_peaks = peaks_result.get("r_peaks", np.array([]))
            comparison = peaks_result.get("comparison")
            ecg_filtered = peaks_result.get("ecg_filtered")
            max_len = len(time_axis)
            peaks = peaks[peaks < max_len]
            r_peaks = r_peaks[r_peaks < max_len]
            
            # Debug info
            st.info(f"Debug: Found {len(r_peaks)} R peaks and {len(peaks)} AO peaks")
            
            # Debug visualization for ECG
            if ecg_filtered is not None:
                with st.expander("🔍 ECG Debug View", expanded=False):
                    fig_debug = go.Figure()
                    fig_debug.add_trace(go.Scatter(
                        x=time_axis,
                        y=ecg_filtered,
                        mode='lines',
                        name='Filtered ECG',
                        line=dict(color='blue', width=1)
                    ))
                    
                    if len(r_peaks) > 0:
                        fig_debug.add_trace(go.Scatter(
                            x=time_axis[r_peaks],
                            y=ecg_filtered[r_peaks],
                            mode='markers',
                            name='Detected R Peaks',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    fig_debug.update_layout(
                        title="Filtered ECG Signal and Detected R Peaks",
                        xaxis_title="Time (s)",
                        yaxis_title="Amplitude",
                        height=300,
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig_debug, use_container_width=True)

            # Display peak counts and heart rates
            col1, col2, col3 = st.columns(3)
            
            if len(peaks) > 1:
                ao_ao_intervals = np.diff(peaks) / fs
                avg_hr_ao = 60 / np.mean(ao_ao_intervals)
                with col1:
                    st.metric("AO Peaks", f"{len(peaks)}")
                with col2:
                    st.metric("AO Heart Rate", f"{avg_hr_ao:.1f} BPM")
            else:
                with col1:
                    st.metric("AO Peaks", f"{len(peaks)}")
                st.warning("Not enough AO peaks detected to calculate heart rate.")
            
            if len(r_peaks) > 1:
                rr_intervals = np.diff(r_peaks) / fs
                avg_hr_ecg = 60 / np.mean(rr_intervals)
                with col3:
                    st.metric("ECG Heart Rate", f"{avg_hr_ecg:.1f} BPM")
            
            # Display comparison statistics if available
            if comparison is not None:
                st.markdown("### 📊 AO-AO vs R-R Interval Comparison")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Correlation", f"{comparison['correlation']:.3f}")
                with col2:
                    st.metric("RMSE", f"{comparison['rmse']:.1f} ms")
                with col3:
                    st.metric("MAE", f"{comparison['mae']:.1f} ms")
                with col4:
                    st.metric("Bias", f"{comparison['mean_diff']:.1f} ms")
                
                # Bland-Altman Plot
                st.markdown("#### Bland-Altman Plot")
                fig_ba = go.Figure()
                
                fig_ba.add_trace(go.Scatter(
                    x=comparison['mean_intervals'],
                    y=comparison['diff_intervals'],
                    mode='markers',
                    marker=dict(size=8, color='blue', opacity=0.6),
                    name='Data points'
                ))
                
                # Mean difference line
                fig_ba.add_hline(
                    y=comparison['mean_diff'],
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Mean: {comparison['mean_diff']:.1f} ms",
                    annotation_position="right"
                )
                
                # Limits of agreement
                upper_loa = comparison['mean_diff'] + 1.96 * comparison['std_diff']
                lower_loa = comparison['mean_diff'] - 1.96 * comparison['std_diff']
                
                fig_ba.add_hline(
                    y=upper_loa,
                    line_dash="dash",
                    line_color="gray",
                    line_width=1.5,
                    annotation_text=f"+1.96 SD: {upper_loa:.1f} ms",
                    annotation_position="right"
                )
                
                fig_ba.add_hline(
                    y=lower_loa,
                    line_dash="dash",
                    line_color="gray",
                    line_width=1.5,
                    annotation_text=f"-1.96 SD: {lower_loa:.1f} ms",
                    annotation_position="right"
                )
                
                fig_ba.update_layout(
                    title="Bland-Altman Plot: AO-AO vs R-R Intervals",
                    xaxis_title="Mean of AO-AO and R-R Intervals (ms)",
                    yaxis_title="Difference (AO-AO - R-R) (ms)",
                    height=500,
                    plot_bgcolor='white',
                    hovermode='closest'
                )
                fig_ba.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                fig_ba.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                
                st.plotly_chart(fig_ba, use_container_width=True)
                
                # Correlation scatter plot
                st.markdown("#### Interval Correlation")
                fig_corr = go.Figure()
                
                fig_corr.add_trace(go.Scatter(
                    x=comparison['rr_intervals_ms'],
                    y=comparison['ao_intervals_ms'],
                    mode='markers',
                    marker=dict(size=8, color='green', opacity=0.6),
                    name='Intervals'
                ))
                
                # Identity line
                min_val = min(np.min(comparison['rr_intervals_ms']), np.min(comparison['ao_intervals_ms']))
                max_val = max(np.max(comparison['rr_intervals_ms']), np.max(comparison['ao_intervals_ms']))
                fig_corr.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='Identity line'
                ))
                
                fig_corr.update_layout(
                    title=f"AO-AO vs R-R Intervals (r = {comparison['correlation']:.3f})",
                    xaxis_title="R-R Interval (ms)",
                    yaxis_title="AO-AO Interval (ms)",
                    height=500,
                    plot_bgcolor='white',
                    hovermode='closest'
                )
                fig_corr.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                fig_corr.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                
                st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("### Peaks Overlayed on Original Signals")
            fig_overlay = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                       subplot_titles=("ECG I with R Peaks and AO Peaks", "Original SCG with AO Peaks"))

            peak_times = time_axis[peaks]

            ecg_segment = signals[start_idx:end_idx, 0]
            fig_overlay.add_trace(go.Scatter(x=time_axis, y=ecg_segment, mode='lines',
                                             line=dict(color='black', width=1), name="ECG I"), row=1, col=1)
            
            # Add R peaks markers
            if len(r_peaks) > 0:
                r_peak_times = time_axis[r_peaks]
                fig_overlay.add_trace(go.Scatter(x=r_peak_times, y=ecg_segment[r_peaks], mode='markers',
                                                 marker=dict(color='green', size=10, symbol='circle'),
                                                 name="R Peaks"), row=1, col=1)
            
            # Add AO peak lines on ECG
            for peak_time in peak_times:
                fig_overlay.add_vline(x=peak_time, line=dict(color='red', width=1.5, dash='dash'),
                                     opacity=0.6, row=1, col=1)

            scg_segment = signals[start_idx:end_idx, 3]
            fig_overlay.add_trace(go.Scatter(x=time_axis, y=scg_segment, mode='lines',
                                             line=dict(color='navy', width=1), name="Original SCG"), row=2, col=1)
            for peak_time in peak_times:
                fig_overlay.add_vline(x=peak_time, line=dict(color='red', width=1.5, dash='dash'),
                                     opacity=0.6, row=2, col=1)

            fig_overlay.add_trace(go.Scatter(x=peak_times, y=peak_times*0, mode='markers',
                                             marker=dict(color='red', size=10, symbol='triangle-down'),
                                             name="AO Peaks", showlegend=True), row=1, col=1)

            fig_overlay.update_layout(height=600, hovermode='x unified', plot_bgcolor='white', margin=dict(t=40, b=40))
            fig_overlay.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)', title_text="Time (s)", row=2, col=1)
            fig_overlay.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)')

            st.plotly_chart(fig_overlay, use_container_width=True)

            st.markdown("### Algorithm Processing Steps")
            fig_final = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                      subplot_titles=("Reconstructed AO Signal", "Seventh Power & Envelope", "Smoothed Envelope & Detected Peaks"))

            fig_final.add_trace(go.Scatter(x=time_axis, y=peaks_result["s_ao_plot"], mode='lines', line=dict(color='black', width=1.5), name="Reconstructed AO"), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=time_axis, y=peaks_result["s_ao_7"], mode='lines', line=dict(color='black', width=1), name="7th Power Signal"), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=time_axis, y=peaks_result["envelope"], mode='lines', line=dict(color='green', width=1.5), name="Envelope"), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=time_axis, y=peaks_result["smoothed_env"], mode='lines', line=dict(color='green', width=1.5), name="Smoothed Env"), row=3, col=1)
            fig_final.add_trace(go.Scatter(x=time_axis[peaks], y=peaks_result["smoothed_env"][peaks], mode='markers', marker=dict(color='red', size=8, symbol='circle'), name="Detected AO Peaks"), row=3, col=1)

            fig_final.update_layout(height=700, hovermode='x unified', plot_bgcolor='white', margin=dict(t=40, b=40))
            fig_final.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)', title_text="Time (s)", row=3, col=1)
            fig_final.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)')

            st.plotly_chart(fig_final, use_container_width=True)

        # Full Record Analysis Section
        if full_record_btn:
            if not use_mti and not is_unhealthy:
                st.warning("Please enable 'Apply Complete Preprocessing' first.")
            else:
                st.divider()
                st.subheader("📊 Full Record Analysis")
                st.info(f"Analyzing entire record ({total_duration:.1f}s) using 10-second windows...")
                
                # Initialize accumulators
                all_ao_peaks = []
                all_r_peaks = []
                all_ao_intervals_times = []
                all_rr_intervals_times = []
                all_ao_intervals = []
                all_rr_intervals = []
                
                window_duration = 10.0  # 10 second windows
                num_windows = int(total_duration / window_duration)
                
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    start_time = time.time()
                    
                    for i in range(num_windows):
                        window_start = i * window_duration
                        window_end = window_start + window_duration
                        
                        status_text.text(f"Processing window {i+1}/{num_windows} ({window_start:.1f}s - {window_end:.1f}s)")
                        progress_bar.progress(int((i+1) / num_windows * 100))
                        
                        # Extract window
                        start_idx_w = int(window_start * fs)
                        end_idx_w = int(window_end * fs)
                        
                        ecg_window = signals[start_idx_w:end_idx_w, 0]
                        scg_window_raw = signals[start_idx_w:end_idx_w, 3]
                        
                        # Apply preprocessing to SCG
                        scg_window_filtered = apply_mti_filter(scg_window_raw, fs) if use_mti else scg_window_raw
                        
                        # Downsample for SVMD
                        q_w = fs / 500
                        if q_w > 1:
                            scg_for_svmd_w = signal.resample(scg_window_filtered, int(len(scg_window_filtered) / q_w))
                            fs_svmd = 500
                        else:
                            scg_for_svmd_w = scg_window_filtered
                            fs_svmd = fs
                        
                        # Run SVMD on window
                        modes_w, omegas_w = svmd(scg_for_svmd_w, max_alpha=svmd_alpha, tau=0, stopc=3)
                        
                        if len(omegas_w) == 0:
                            continue
                        
                        # Select AO modes
                        s_ao_svmd_w, wfs_w, wf_mean_w, selected_idx_w = select_ao_modes(
                            modes_w, omegas_w, fs_svmd
                        )
                        
                        # Upsample back to original sampling rate
                        if q_w > 1:
                            s_ao_w = signal.resample(s_ao_svmd_w, len(scg_window_filtered))
                        else:
                            s_ao_w = s_ao_svmd_w
                        
                        # Extract AO peaks
                        s_ao_7_w, envelope_w, smoothed_env_w, ao_peaks_w = extract_ao_peaks(
                            s_ao_w, fs, prominence_factor, power=power_exp
                        )
                        
                        # Detect R peaks from ECG
                        r_peaks_w, ecg_filtered_w, ecg_integrated_w = detect_r_peaks(ecg_window, fs)
                        
                        # Adjust peak indices to global time and collect
                        if len(ao_peaks_w) > 0:
                            ao_peaks_global = ao_peaks_w + start_idx_w
                            all_ao_peaks.extend(ao_peaks_global)
                            
                            # Calculate intervals for this window
                            if len(ao_peaks_w) > 1:
                                ao_intervals_w = np.diff(ao_peaks_w) / fs * 1000
                                ao_interval_times_w = (ao_peaks_w[:-1] + ao_peaks_w[1:]) / 2 / fs + window_start
                                all_ao_intervals.extend(ao_intervals_w)
                                all_ao_intervals_times.extend(ao_interval_times_w)
                        
                        if len(r_peaks_w) > 0:
                            r_peaks_global = r_peaks_w + start_idx_w
                            all_r_peaks.extend(r_peaks_global)
                            
                            # Calculate intervals for this window
                            if len(r_peaks_w) > 1:
                                rr_intervals_w = np.diff(r_peaks_w) / fs * 1000
                                rr_interval_times_w = (r_peaks_w[:-1] + r_peaks_w[1:]) / 2 / fs + window_start
                                all_rr_intervals.extend(rr_intervals_w)
                                all_rr_intervals_times.extend(rr_interval_times_w)
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    status_text.text(f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds")
                    progress_bar.progress(100)
                    
                    # Convert to numpy arrays
                    all_ao_peaks = np.array(all_ao_peaks)
                    all_r_peaks = np.array(all_r_peaks)
                    all_ao_intervals = np.array(all_ao_intervals)
                    all_rr_intervals = np.array(all_rr_intervals)
                    all_ao_intervals_times = np.array(all_ao_intervals_times)
                    all_rr_intervals_times = np.array(all_rr_intervals_times)
                    
                    st.success(f"✅ Found {len(all_ao_peaks)} AO peaks and {len(all_r_peaks)} R peaks across entire record | ⏱️ Processing time: {elapsed_time:.2f} seconds")

                    if save_json_output and len(all_ao_peaks) > 0:
                        saved_file = save_peaks_to_json(all_ao_peaks, fs, selected_record, output_folder)
                        st.info(f"💾 AO Peaks saved locally to: {saved_file}")
                    
                    # Calculate comparison statistics
                    if len(all_ao_peaks) > 1 and len(all_r_peaks) > 1:
                        # First match intervals to same length
                        min_len = min(len(all_ao_intervals), len(all_rr_intervals))
                        all_ao_intervals = all_ao_intervals[:min_len]
                        all_rr_intervals = all_rr_intervals[:min_len]
                        all_ao_intervals_times = all_ao_intervals_times[:min_len]
                        all_rr_intervals_times = all_rr_intervals_times[:min_len]
                        
                        # Apply paired outlier removal if enabled
                        if remove_outliers:
                            original_count = len(all_ao_intervals)
                            
                            # Use IQR method on both interval sets to identify outliers
                            def get_outlier_mask(intervals):
                                if len(intervals) < 4:
                                    return np.ones(len(intervals), dtype=bool)
                                q1 = np.percentile(intervals, 25)
                                q3 = np.percentile(intervals, 75)
                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                return (intervals >= lower_bound) & (intervals <= upper_bound)
                            
                            # Get masks for both interval sets
                            ao_mask = get_outlier_mask(all_ao_intervals)
                            rr_mask = get_outlier_mask(all_rr_intervals)
                            
                            # Combined mask: keep only pairs where BOTH are inliers
                            combined_mask = ao_mask & rr_mask
                            
                            # Apply mask to all interval arrays
                            all_ao_intervals = all_ao_intervals[combined_mask]
                            all_rr_intervals = all_rr_intervals[combined_mask]
                            all_ao_intervals_times = all_ao_intervals_times[combined_mask]
                            all_rr_intervals_times = all_rr_intervals_times[combined_mask]
                            
                            pairs_removed = original_count - len(all_ao_intervals)
                            
                            st.info(f"🔧 Outlier removal: {pairs_removed} interval pairs removed (maintained equal counts)")
                        
                        # Use the (potentially filtered) intervals directly
                        ao_intervals_matched = all_ao_intervals
                        rr_intervals_matched = all_rr_intervals
                        
                        # Calculate statistics
                        correlation = np.corrcoef(ao_intervals_matched, rr_intervals_matched)[0, 1]
                        rmse = np.sqrt(np.mean((ao_intervals_matched - rr_intervals_matched)**2))
                        mae = np.mean(np.abs(ao_intervals_matched - rr_intervals_matched))
                        
                        # Bland-Altman statistics
                        mean_intervals = (ao_intervals_matched + rr_intervals_matched) / 2
                        diff_intervals = ao_intervals_matched - rr_intervals_matched
                        mean_diff = np.mean(diff_intervals)
                        std_diff = np.std(diff_intervals)
                        
                        # Display metrics
                        st.markdown("#### 📈 Interval Comparison Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Correlation (r)", f"{correlation:.3f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.1f} ms")
                        with col3:
                            st.metric("MAE", f"{mae:.1f} ms")
                        with col4:
                            st.metric("Bias", f"{mean_diff:.1f} ms")
                        
                        # Time series plot with overlayed intervals
                        st.markdown("#### ⏱️ Signals and Interval Time Series Comparison")
                        
                        # Create subplot with 3 rows: ECG, SCG, and Intervals
                        fig_intervals = make_subplots(
                            rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=(
                                "ECG I with R Peaks", 
                                "SCG with AO Peaks", 
                                "R-R and AO-AO Intervals"
                            ),
                            row_heights=[0.3, 0.3, 0.4]
                        )
                        
                        # Prepare full record signals and time axis
                        full_ecg = signals[:, 0]
                        full_scg = signals[:, 3]
                        full_time_axis = np.arange(len(full_ecg)) / fs
                        
                        # Row 1: ECG signal with R peaks
                        fig_intervals.add_trace(go.Scatter(
                            x=full_time_axis,
                            y=full_ecg,
                            mode='lines',
                            line=dict(color='black', width=0.5),
                            name='ECG I',
                            showlegend=True
                        ), row=1, col=1)
                        
                        if len(all_r_peaks) > 0:
                            r_peak_times = all_r_peaks / fs
                            fig_intervals.add_trace(go.Scatter(
                                x=r_peak_times,
                                y=full_ecg[all_r_peaks.astype(int)],
                                mode='markers',
                                marker=dict(color='green', size=4, symbol='circle'),
                                name='R Peaks',
                                showlegend=True
                            ), row=1, col=1)
                        
                        # Row 2: SCG signal with AO peaks
                        fig_intervals.add_trace(go.Scatter(
                            x=full_time_axis,
                            y=full_scg,
                            mode='lines',
                            line=dict(color='navy', width=0.5),
                            name='SCG',
                            showlegend=True
                        ), row=2, col=1)
                        
                        if len(all_ao_peaks) > 0:
                            ao_peak_times = all_ao_peaks / fs
                            fig_intervals.add_trace(go.Scatter(
                                x=ao_peak_times,
                                y=full_scg[all_ao_peaks.astype(int)],
                                mode='markers',
                                marker=dict(color='red', size=4, symbol='circle'),
                                name='AO Peaks',
                                showlegend=True
                            ), row=2, col=1)
                        
                        # Row 3: Interval comparison
                        fig_intervals.add_trace(go.Scatter(
                            x=all_rr_intervals_times,
                            y=all_rr_intervals,
                            mode='lines+markers',
                            line=dict(color='green', width=1.5),
                            marker=dict(size=4),
                            name='R-R Intervals',
                            showlegend=True
                        ), row=3, col=1)
                        
                        fig_intervals.add_trace(go.Scatter(
                            x=all_ao_intervals_times,
                            y=all_ao_intervals,
                            mode='lines+markers',
                            line=dict(color='red', width=1.5),
                            marker=dict(size=4),
                            name='AO-AO Intervals',
                            showlegend=True
                        ), row=3, col=1)
                        
                        # Update layout
                        fig_intervals.update_layout(
                            title=f"Full Record Analysis ({total_duration:.1f}s)",
                            height=900,
                            plot_bgcolor='white',
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(x=1.02, y=0.5, bgcolor='rgba(255,255,255,0.8)')
                        )
                        
                        # Update axes
                        fig_intervals.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                        fig_intervals.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 0.3)')
                        fig_intervals.update_xaxes(title_text="Time (s)", row=3, col=1)
                        fig_intervals.update_yaxes(title_text="ECG (mV)", row=1, col=1)
                        fig_intervals.update_yaxes(title_text="SCG (m/s²)", row=2, col=1)
                        fig_intervals.update_yaxes(title_text="Interval (ms)", row=3, col=1)
                        
                        st.plotly_chart(fig_intervals, use_container_width=True)
                        
                        # Bland-Altman Plot
                        st.markdown("#### 📊 Bland-Altman Plot")
                        fig_ba_full = go.Figure()
                        
                        fig_ba_full.add_trace(go.Scatter(
                            x=mean_intervals,
                            y=diff_intervals,
                            mode='markers',
                            marker=dict(size=6, color='blue', opacity=0.5),
                            name='Data points'
                        ))
                        
                        # Mean difference line
                        fig_ba_full.add_hline(
                            y=mean_diff,
                            line_dash="solid",
                            line_color="red",
                            line_width=2,
                            annotation_text=f"Mean: {mean_diff:.1f} ms",
                            annotation_position="right"
                        )
                        
                        # Limits of agreement
                        upper_loa_full = mean_diff + 1.96 * std_diff
                        lower_loa_full = mean_diff - 1.96 * std_diff
                        
                        fig_ba_full.add_hline(
                            y=upper_loa_full,
                            line_dash="dash",
                            line_color="gray",
                            line_width=1.5,
                            annotation_text=f"+1.96 SD: {upper_loa_full:.1f} ms",
                            annotation_position="right"
                        )
                        
                        fig_ba_full.add_hline(
                            y=lower_loa_full,
                            line_dash="dash",
                            line_color="gray",
                            line_width=1.5,
                            annotation_text=f"-1.96 SD: {lower_loa_full:.1f} ms",
                            annotation_position="right"
                        )
                        
                        fig_ba_full.update_layout(
                            title="Bland-Altman Plot: AO-AO vs R-R Intervals (Full Record)",
                            xaxis_title="Mean of AO-AO and R-R Intervals (ms)",
                            yaxis_title="Difference (AO-AO - R-R) (ms)",
                            height=500,
                            plot_bgcolor='white',
                            hovermode='closest'
                        )
                        fig_ba_full.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                        fig_ba_full.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 0.3)')
                        
                        st.plotly_chart(fig_ba_full, use_container_width=True)
                        
                        # Correlation scatter plot
                        st.markdown("#### 🔗 Interval Correlation")
                        fig_corr_full = go.Figure()
                        
                        fig_corr_full.add_trace(go.Scatter(
                            x=rr_intervals_matched,
                            y=ao_intervals_matched,
                            mode='markers',
                            marker=dict(size=6, color='purple', opacity=0.5),
                            name='Intervals'
                        ))
                        
                        # Identity line
                        min_val_full = min(np.min(rr_intervals_matched), 
                                          np.min(ao_intervals_matched))
                        max_val_full = max(np.max(rr_intervals_matched), 
                                          np.max(ao_intervals_matched))
                        fig_corr_full.add_trace(go.Scatter(
                            x=[min_val_full, max_val_full],
                            y=[min_val_full, max_val_full],
                            mode='lines',
                            line=dict(color='red', dash='dash', width=2),
                            name='Identity line'
                        ))
                        
                        fig_corr_full.update_layout(
                            title=f"AO-AO vs R-R Intervals - Full Record (r = {correlation:.3f})",
                            xaxis_title="R-R Interval (ms)",
                            yaxis_title="AO-AO Interval (ms)",
                            height=500,
                            plot_bgcolor='white',
                            hovermode='closest'
                        )
                        fig_corr_full.update_xaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                        fig_corr_full.update_yaxes(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)')
                        
                        st.plotly_chart(fig_corr_full, use_container_width=True)
                        
                    else:
                        st.warning("Not enough peaks detected for interval comparison.")
                
                except Exception as e:
                    st.error(f"Error processing full record: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if batch_random_btn:
            if is_unhealthy:
                st.info("Batch analysis is available for CEBS only.")
            elif not use_mti:
                st.warning("Please enable 'Apply Complete Preprocessing' first.")
            else:
                batch_results = []
                status_text = st.empty()
                progress_bar = st.progress(0)
                total_records = len(records)
                with st.spinner("Running random 10s batch for all records..."):
                    for idx, record_name in enumerate(records, start=1):
                        status_text.text(f"Processing: {record_name} ({idx}/{total_records})")
                        progress_bar.progress(int((idx / total_records) * 100))
                        try:
                            header = wfdb.rdheader(os.path.join(db_path, record_name))
                            fs_r = header.fs
                            total_duration = header.sig_len / fs_r
                            if total_duration <= 70:
                                batch_results.append({
                                    "Record": record_name,
                                    "Start Time (s)": "N/A",
                                    "Peaks": "N/A",
                                    "Heart Rate (BPM)": "N/A",
                                    "IMFs": "N/A",
                                    "Status": "Skipped (duration <= 70s)"
                                })
                                continue

                            start_time_r = np.random.uniform(60, total_duration - 10)
                            start_idx_r = int(start_time_r * fs_r)
                            end_idx_r = int((start_time_r + 10) * fs_r)

                            record_r = wfdb.rdsamp(os.path.join(db_path, record_name))
                            signals_r = record_r[0]

                            scg_filtered_full_r = apply_mti_filter(signals_r[:, 3], fs_r) if use_mti else signals_r[:, 3]
                            scg_segment_r = scg_filtered_full_r[start_idx_r:end_idx_r]

                            q_r = fs_r / 500
                            if q_r > 1:
                                scg_for_svmd_r = signal.resample(scg_segment_r, int(len(scg_segment_r) / q_r))
                                fs_svmd_r = 500
                            else:
                                scg_for_svmd_r = scg_segment_r
                                fs_svmd_r = fs_r

                            modes_r, omegas_r = svmd(scg_for_svmd_r, max_alpha=svmd_alpha, tau=0, stopc=3)
                            if len(omegas_r) == 0:
                                batch_results.append({
                                    "Record": record_name,
                                    "Start Time (s)": f"{start_time_r:.2f}",
                                    "Peaks": "N/A",
                                    "Heart Rate (BPM)": "N/A",
                                    "IMFs": "0",
                                    "Status": "SVMD returned no modes"
                                })
                                continue

                            s_ao_svmd_r, wfs_r, wf_mean_r, selected_idx_r = select_ao_modes(modes_r, omegas_r, fs_svmd_r)
                            if q_r > 1:
                                s_ao_plot_r = signal.resample(s_ao_svmd_r, len(scg_segment_r))
                            else:
                                s_ao_plot_r = s_ao_svmd_r

                            s_ao_7_r, envelope_r, smoothed_env_r, peaks_r = extract_ao_peaks(
                                s_ao_plot_r, fs_r, prominence_factor, power=power_exp
                            )

                            if len(peaks_r) > 1:
                                ao_ao_intervals = np.diff(peaks_r) / fs_r
                                avg_hr = 60 / np.mean(ao_ao_intervals)
                                hr_text = f"{avg_hr:.1f}"
                            else:
                                hr_text = "N/A"

                            batch_results.append({
                                "Record": record_name,
                                "Start Time (s)": f"{start_time_r:.2f}",
                                "Peaks": f"{len(peaks_r)}",
                                "Heart Rate (BPM)": hr_text,
                                "IMFs": f"{len(selected_idx_r)}/{len(modes_r)}",
                                "Status": "OK"
                            })

                        except Exception as e:
                            batch_results.append({
                                "Record": record_name,
                                "Start Time (s)": "N/A",
                                "Peaks": "N/A",
                                "Heart Rate (BPM)": "N/A",
                                "IMFs": "N/A",
                                "Status": f"Error: {e}"
                            })

                status_text.text("Batch processing complete.")
                progress_bar.progress(100)
                st.session_state["batch_random_results"] = batch_results

        if "batch_random_results" in st.session_state:
            st.divider()
            st.subheader("Random 10s Batch Results (Start >= 60s)")
            batch_results = st.session_state["batch_random_results"]
            st.dataframe(batch_results, use_container_width=True, hide_index=True)

            for result in batch_results:
                expander_title = f"{result['Record']} - {result['Status']}"
                with st.expander(expander_title, expanded=False):
                    st.write(f"**Start Time (s):** {result['Start Time (s)']}")
                    st.write(f"**Peaks:** {result['Peaks']}")
                    st.write(f"**Heart Rate (BPM):** {result['Heart Rate (BPM)']}")
                    st.write(f"**IMFs:** {result['IMFs']}")

        # Batch Full Record Analysis
        if batch_full_record_btn:
            if not use_mti:
                st.warning("Please enable 'Apply Complete Preprocessing' first.")
            else:
                st.divider()
                st.subheader("🔄 Batch Full Record Analysis")
                st.info(f"Analyzing full records for all {len(records)} records using 10-second windows...")

                batch_full_results = []
                status_text = st.empty()
                progress_bar = st.progress(0)

                try:
                    if is_unhealthy:
                        labels_df = load_unhealthy_labels(base_dir)
                        skip_seconds = unhealthy_skip_seconds or 0.0

                        for idx, record_name in enumerate(records, start=1):
                            status_text.text(f"Processing: {record_name} ({idx}/{len(records)})")
                            progress_bar.progress(int((idx / len(records)) * 100))

                            try:
                                signals_batch, _, fs_batch = load_unhealthy_patient_data(record_name, base_dir)
                                total_duration_batch = len(signals_batch) / fs_batch
                                label_str = get_unhealthy_label_str(labels_df, record_name)

                                window_duration = 10.0
                                if total_duration_batch <= skip_seconds + window_duration:
                                    batch_full_results.append({
                                        "Record": record_name,
                                        "Condition": label_str,
                                        "Duration (s)": f"{total_duration_batch:.1f}",
                                        "AO Peaks": "N/A",
                                        "R Peaks": "N/A",
                                        "Intervals": "N/A",
                                        "Correlation": "N/A",
                                        "RMSE (ms)": "N/A",
                                        "MAE (ms)": "N/A",
                                        "Status": "⚠ Skipped (duration too short)"
                                    })
                                    continue

                                # Initialize accumulators
                                all_ao_intervals_batch = []
                                all_rr_intervals_batch = []
                                all_ao_peaks_batch = []
                                all_r_peaks_batch = []

                                usable_duration = total_duration_batch - skip_seconds
                                num_windows = int(usable_duration / window_duration)
                                base_start_idx = int(skip_seconds * fs_batch)

                                for i in range(num_windows):
                                    window_start = skip_seconds + i * window_duration
                                    window_end = window_start + window_duration

                                    start_idx_w = int(window_start * fs_batch)
                                    end_idx_w = int(window_end * fs_batch)
                                    if end_idx_w > len(signals_batch):
                                        break

                                    ecg_window = signals_batch[start_idx_w:end_idx_w, 0]
                                    scg_window_raw = signals_batch[start_idx_w:end_idx_w, 3]

                                    scg_window_filtered = apply_mti_filter(scg_window_raw, fs_batch) if use_mti else scg_window_raw

                                    q_w = fs_batch / 500
                                    if q_w > 1:
                                        scg_for_svmd_w = signal.resample(scg_window_filtered, int(len(scg_window_filtered) / q_w))
                                        fs_svmd = 500
                                    else:
                                        scg_for_svmd_w = scg_window_filtered
                                        fs_svmd = fs_batch

                                    modes_w, omegas_w = svmd(scg_for_svmd_w, max_alpha=svmd_alpha, tau=0, stopc=3)
                                    if len(omegas_w) == 0:
                                        continue

                                    s_ao_svmd_w, _, _, _ = select_ao_modes(modes_w, omegas_w, fs_svmd)
                                    if q_w > 1:
                                        s_ao_w = signal.resample(s_ao_svmd_w, len(scg_window_filtered))
                                    else:
                                        s_ao_w = s_ao_svmd_w

                                    _, _, _, ao_peaks_w = extract_ao_peaks(
                                        s_ao_w, fs_batch, prominence_factor, power=power_exp
                                    )
                                    r_peaks_w, _, _ = detect_r_peaks(ecg_window, fs_batch)

                                    if len(ao_peaks_w) > 1:
                                        ao_intervals_w = np.diff(ao_peaks_w) / fs_batch * 1000
                                        all_ao_intervals_batch.extend(ao_intervals_w)
                                    if len(r_peaks_w) > 1:
                                        rr_intervals_w = np.diff(r_peaks_w) / fs_batch * 1000
                                        all_rr_intervals_batch.extend(rr_intervals_w)

                                    if len(ao_peaks_w) > 0:
                                        all_ao_peaks_batch.extend(ao_peaks_w + start_idx_w)
                                    if len(r_peaks_w) > 0:
                                        all_r_peaks_batch.extend(r_peaks_w + start_idx_w)

                                all_ao_intervals_batch = np.array(all_ao_intervals_batch)
                                all_rr_intervals_batch = np.array(all_rr_intervals_batch)

                                if remove_outliers:
                                    min_len = min(len(all_ao_intervals_batch), len(all_rr_intervals_batch))
                                    all_ao_intervals_batch = all_ao_intervals_batch[:min_len]
                                    all_rr_intervals_batch = all_rr_intervals_batch[:min_len]

                                    def get_outlier_mask(intervals):
                                        if len(intervals) < 4:
                                            return np.ones(len(intervals), dtype=bool)
                                        q1 = np.percentile(intervals, 25)
                                        q3 = np.percentile(intervals, 75)
                                        iqr = q3 - q1
                                        lower_bound = q1 - 1.5 * iqr
                                        upper_bound = q3 + 1.5 * iqr
                                        return (intervals >= lower_bound) & (intervals <= upper_bound)

                                    ao_mask = get_outlier_mask(all_ao_intervals_batch)
                                    rr_mask = get_outlier_mask(all_rr_intervals_batch)
                                    combined_mask = ao_mask & rr_mask

                                    all_ao_intervals_batch = all_ao_intervals_batch[combined_mask]
                                    all_rr_intervals_batch = all_rr_intervals_batch[combined_mask]

                                min_len = min(len(all_ao_intervals_batch), len(all_rr_intervals_batch))
                                if min_len > 1:
                                    ao_intervals_matched = all_ao_intervals_batch[:min_len]
                                    rr_intervals_matched = all_rr_intervals_batch[:min_len]

                                    correlation = np.corrcoef(ao_intervals_matched, rr_intervals_matched)[0, 1]
                                    rmse = np.sqrt(np.mean((ao_intervals_matched - rr_intervals_matched)**2))
                                    mae = np.mean(np.abs(ao_intervals_matched - rr_intervals_matched))

                                    if save_json_output and len(all_ao_peaks_batch) > 0:
                                        save_peaks_to_json(np.array(all_ao_peaks_batch), fs_batch, record_name, output_folder)

                                    batch_full_results.append({
                                        "Record": record_name,
                                        "Condition": label_str,
                                        "Duration (s)": f"{total_duration_batch:.1f}",
                                        "AO Peaks": len(all_ao_peaks_batch),
                                        "R Peaks": len(all_r_peaks_batch),
                                        "Intervals": len(ao_intervals_matched),
                                        "Correlation": f"{correlation:.3f}",
                                        "RMSE (ms)": f"{rmse:.1f}",
                                        "MAE (ms)": f"{mae:.1f}",
                                        "Status": "✓ OK"
                                    })
                                else:
                                    batch_full_results.append({
                                        "Record": record_name,
                                        "Condition": label_str,
                                        "Duration (s)": f"{total_duration_batch:.1f}",
                                        "AO Peaks": len(all_ao_peaks_batch),
                                        "R Peaks": len(all_r_peaks_batch),
                                        "Intervals": "N/A",
                                        "Correlation": "N/A",
                                        "RMSE (ms)": "N/A",
                                        "MAE (ms)": "N/A",
                                        "Status": "⚠ Insufficient intervals"
                                    })

                            except Exception as e:
                                batch_full_results.append({
                                    "Record": record_name,
                                    "Condition": "Unknown",
                                    "Duration (s)": "N/A",
                                    "AO Peaks": "N/A",
                                    "R Peaks": "N/A",
                                    "Intervals": "N/A",
                                    "Correlation": "N/A",
                                    "RMSE (ms)": "N/A",
                                    "MAE (ms)": "N/A",
                                    "Status": "❌ Error"
                                })

                    else:
                        for idx, record_name in enumerate(records, start=1):
                            status_text.text(f"Processing: {record_name} ({idx}/{len(records)})")
                            progress_bar.progress(int((idx / len(records)) * 100))

                            try:
                                # Load record
                                header = wfdb.rdheader(os.path.join(db_path, record_name))
                                fs_batch = header.fs
                                total_duration_batch = header.sig_len / fs_batch

                                record_data = wfdb.rdsamp(os.path.join(db_path, record_name))
                                signals_batch = record_data[0]

                                # Initialize accumulators
                                all_ao_intervals_batch = []
                                all_rr_intervals_batch = []
                                all_ao_peaks_batch = []
                                all_r_peaks_batch = []

                                window_duration = 10.0
                                num_windows = int(total_duration_batch / window_duration)

                                # Process each 10s window
                                for i in range(num_windows):
                                    window_start = i * window_duration
                                    window_end = window_start + window_duration

                                    start_idx_w = int(window_start * fs_batch)
                                    end_idx_w = int(window_end * fs_batch)

                                    ecg_window = signals_batch[start_idx_w:end_idx_w, 0]
                                    scg_window_raw = signals_batch[start_idx_w:end_idx_w, 3]

                                    # Apply preprocessing
                                    scg_window_filtered = apply_mti_filter(scg_window_raw, fs_batch) if use_mti else scg_window_raw

                                    # Downsample for SVMD
                                    q_w = fs_batch / 500
                                    if q_w > 1:
                                        scg_for_svmd_w = signal.resample(scg_window_filtered, int(len(scg_window_filtered) / q_w))
                                        fs_svmd = 500
                                    else:
                                        scg_for_svmd_w = scg_window_filtered
                                        fs_svmd = fs_batch

                                    # Run SVMD
                                    modes_w, omegas_w = svmd(scg_for_svmd_w, max_alpha=svmd_alpha, tau=0, stopc=3)

                                    if len(omegas_w) == 0:
                                        continue

                                    # Select AO modes
                                    s_ao_svmd_w, _, _, _ = select_ao_modes(modes_w, omegas_w, fs_svmd)

                                    # Upsample
                                    if q_w > 1:
                                        s_ao_w = signal.resample(s_ao_svmd_w, len(scg_window_filtered))
                                    else:
                                        s_ao_w = s_ao_svmd_w

                                    # Extract peaks
                                    _, _, _, ao_peaks_w = extract_ao_peaks(
                                        s_ao_w, fs_batch, prominence_factor, power=power_exp
                                    )
                                    r_peaks_w, _, _ = detect_r_peaks(ecg_window, fs_batch)

                                    # Collect peaks and intervals
                                    if len(ao_peaks_w) > 1:
                                        ao_intervals_w = np.diff(ao_peaks_w) / fs_batch * 1000
                                        all_ao_intervals_batch.extend(ao_intervals_w)
                                    if len(r_peaks_w) > 1:
                                        rr_intervals_w = np.diff(r_peaks_w) / fs_batch * 1000
                                        all_rr_intervals_batch.extend(rr_intervals_w)

                                    if len(ao_peaks_w) > 0:
                                        all_ao_peaks_batch.extend(ao_peaks_w + start_idx_w)
                                    if len(r_peaks_w) > 0:
                                        all_r_peaks_batch.extend(r_peaks_w + start_idx_w)

                                # Convert to arrays
                                all_ao_intervals_batch = np.array(all_ao_intervals_batch)
                                all_rr_intervals_batch = np.array(all_rr_intervals_batch)

                                # Apply paired outlier removal if enabled
                                if remove_outliers:
                                    min_len = min(len(all_ao_intervals_batch), len(all_rr_intervals_batch))
                                    all_ao_intervals_batch = all_ao_intervals_batch[:min_len]
                                    all_rr_intervals_batch = all_rr_intervals_batch[:min_len]

                                    def get_outlier_mask(intervals):
                                        if len(intervals) < 4:
                                            return np.ones(len(intervals), dtype=bool)
                                        q1 = np.percentile(intervals, 25)
                                        q3 = np.percentile(intervals, 75)
                                        iqr = q3 - q1
                                        lower_bound = q1 - 1.5 * iqr
                                        upper_bound = q3 + 1.5 * iqr
                                        return (intervals >= lower_bound) & (intervals <= upper_bound)

                                    ao_mask = get_outlier_mask(all_ao_intervals_batch)
                                    rr_mask = get_outlier_mask(all_rr_intervals_batch)
                                    combined_mask = ao_mask & rr_mask

                                    all_ao_intervals_batch = all_ao_intervals_batch[combined_mask]
                                    all_rr_intervals_batch = all_rr_intervals_batch[combined_mask]

                                # Match lengths
                                min_len = min(len(all_ao_intervals_batch), len(all_rr_intervals_batch))
                                if min_len > 1:
                                    ao_intervals_matched = all_ao_intervals_batch[:min_len]
                                    rr_intervals_matched = all_rr_intervals_batch[:min_len]

                                    # Compute statistics
                                    correlation = np.corrcoef(ao_intervals_matched, rr_intervals_matched)[0, 1]
                                    rmse = np.sqrt(np.mean((ao_intervals_matched - rr_intervals_matched)**2))
                                    mae = np.mean(np.abs(ao_intervals_matched - rr_intervals_matched))

                                    if save_json_output and len(all_ao_peaks_batch) > 0:
                                        save_peaks_to_json(np.array(all_ao_peaks_batch), fs_batch, record_name, output_folder)

                                    batch_full_results.append({
                                        "Record": record_name,
                                        "Duration (s)": f"{total_duration_batch:.1f}",
                                        "AO Peaks": len(all_ao_peaks_batch),
                                        "R Peaks": len(all_r_peaks_batch),
                                        "Intervals": len(ao_intervals_matched),
                                        "Correlation": f"{correlation:.3f}",
                                        "RMSE (ms)": f"{rmse:.1f}",
                                        "MAE (ms)": f"{mae:.1f}",
                                        "Status": "✓ OK"
                                    })
                                else:
                                    batch_full_results.append({
                                        "Record": record_name,
                                        "Duration (s)": f"{total_duration_batch:.1f}",
                                        "AO Peaks": len(all_ao_peaks_batch),
                                        "R Peaks": len(all_r_peaks_batch),
                                        "Intervals": "N/A",
                                        "Correlation": "N/A",
                                        "RMSE (ms)": "N/A",
                                        "MAE (ms)": "N/A",
                                        "Status": "⚠ Insufficient intervals"
                                    })

                            except Exception as e:
                                batch_full_results.append({
                                    "Record": record_name,
                                    "Duration (s)": "N/A",
                                    "AO Peaks": "N/A",
                                    "R Peaks": "N/A",
                                    "Intervals": "N/A",
                                    "Correlation": "N/A",
                                    "RMSE (ms)": "N/A",
                                    "MAE (ms)": "N/A",
                                    "Status": "❌ Error"
                                })

                    status_text.text("Batch processing complete!")
                    progress_bar.progress(100)

                    # Display results
                    st.divider()
                    st.subheader("📋 Full Record Batch Results")
                    st.dataframe(batch_full_results, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Error in batch full record analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Error processing record: {e}")