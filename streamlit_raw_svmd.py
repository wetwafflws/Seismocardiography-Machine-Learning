from pathlib import Path
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import hilbert
from scipy.stats import kurtosis


REQUIRED_COLS = ["timestamp_ms", "x_g", "y_g", "z_g", "beat_event"]
OPTIONAL_PPG_COL = "ppg_raw"
DEFAULT_SEARCH_DIR = "SUBJECT_Data"
META_SUFFIX = "_meta.json"


st.set_page_config(page_title="Raw SCG SVMD", layout="wide")
st.title("Raw SCG SVMD")
st.caption("Run SVMD on your raw SCG CSV with PPG beat references.")


def _read_csv_from_source(uploaded_file, selected_path: str) -> pd.DataFrame:
    read_kwargs = {
        "comment": "#",
        "engine": "python",
    }
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, **read_kwargs)
    return pd.read_csv(selected_path, **read_kwargs)


def _resolve_meta_path(csv_path: str) -> Path:
    csv_file = Path(csv_path)
    return csv_file.with_name(f"{csv_file.stem}{META_SUFFIX}")


def _read_metadata_for_csv(csv_path: str) -> tuple[Optional[dict], str]:
    if not csv_path:
        return None, ""
    meta_path = _resolve_meta_path(csv_path)
    if not meta_path.exists():
        return None, str(meta_path)
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), str(meta_path)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
    out["x_g"] = pd.to_numeric(out["x_g"], errors="coerce")
    out["y_g"] = pd.to_numeric(out["y_g"], errors="coerce")
    out["z_g"] = pd.to_numeric(out["z_g"], errors="coerce")
    out["beat_event"] = pd.to_numeric(out["beat_event"], errors="coerce").fillna(0).astype(int)
    if OPTIONAL_PPG_COL in out.columns:
        out[OPTIONAL_PPG_COL] = pd.to_numeric(out[OPTIONAL_PPG_COL], errors="coerce")

    out = out.dropna(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return out


def _compute_rate(scg_df: pd.DataFrame) -> tuple[float, float]:
    if len(scg_df) < 2:
        return 0.0, 0.0
    duration_s = (scg_df["timestamp_ms"].iloc[-1] - scg_df["timestamp_ms"].iloc[0]) / 1000.0
    if duration_s <= 0:
        return 0.0, 0.0
    actual_hz = (len(scg_df) - 1) / duration_s
    return duration_s, actual_hz


def _compute_rate_from_ts(ts_ms: np.ndarray) -> tuple[float, float]:
    if len(ts_ms) < 2:
        return 0.0, 0.0
    duration_s = (float(ts_ms[-1]) - float(ts_ms[0])) / 1000.0
    if duration_s <= 0:
        return 0.0, 0.0
    actual_hz = (len(ts_ms) - 1) / duration_s
    return duration_s, actual_hz


def _bandpass_ppg(ppg_signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    if fs <= 0:
        return ppg_signal
    nyq = 0.5 * fs
    low = max(0.01, low_hz) / nyq
    high = min(0.99 * nyq, high_hz) / nyq
    if low >= high:
        return ppg_signal
    sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, ppg_signal)


def _detect_ppg_peaks(ppg_signal: np.ndarray, fs: float, max_bpm: float, prominence_factor: float) -> np.ndarray:
    if len(ppg_signal) < 3 or fs <= 0:
        return np.array([], dtype=int)
        
    min_distance = int(max(1, (60.0 / max(max_bpm, 1.0)) * fs))
    
    sig_range = np.percentile(ppg_signal, 95) - np.percentile(ppg_signal, 5)
    dynamic_prom = sig_range * prominence_factor if sig_range > 0 else prominence_factor
    
    peaks, _ = signal.find_peaks(ppg_signal, distance=min_distance, prominence=dynamic_prom)
    return peaks.astype(int)


def format_timestamp(seconds_val: float) -> str:
    hours = int(seconds_val // 3600)
    minutes = int((seconds_val % 3600) // 60)
    seconds = seconds_val % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:07.4f}"


def _build_peaks_payload(peak_seconds, record_name: str, peak_label: str) -> dict:
    timestamps = [format_timestamp(float(sec)) for sec in peak_seconds]
    return {f"{record_name}_{peak_label}_Peaks": timestamps}


def save_peaks_to_json(peaks_indices, fs, record_name, output_dir="Saved_Peaks", peak_label="AO") -> str:
    os.makedirs(output_dir, exist_ok=True)
    fs_safe = float(fs) if fs else 0.0
    peak_seconds = np.asarray(peaks_indices, dtype=float) / fs_safe if fs_safe > 0 else []
    data = _build_peaks_payload(peak_seconds, record_name, peak_label)

    out_path = os.path.join(output_dir, f"{record_name}_{peak_label}_Peaks.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

    return out_path


def save_peaks_seconds_to_json(peak_seconds, record_name, output_dir="Saved_Peaks", peak_label="AO") -> str:
    os.makedirs(output_dir, exist_ok=True)
    data = _build_peaks_payload(peak_seconds, record_name, peak_label)

    out_path = os.path.join(output_dir, f"{record_name}_{peak_label}_Peaks.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

    return out_path


def _map_times_to_indices(time_axis_s: np.ndarray, event_times_s: np.ndarray) -> np.ndarray:
    if len(time_axis_s) == 0 or len(event_times_s) == 0:
        return np.array([], dtype=int)
    idx = np.searchsorted(time_axis_s, event_times_s)
    idx = np.clip(idx, 1, len(time_axis_s) - 1)
    prev_idx = idx - 1
    next_idx = idx
    prev_dt = np.abs(time_axis_s[prev_idx] - event_times_s)
    next_dt = np.abs(time_axis_s[next_idx] - event_times_s)
    chosen = np.where(next_dt < prev_dt, next_idx, prev_idx)
    return chosen.astype(int)


def resample_for_processing(raw_signal, fs_original, target_fs=500):
    if fs_original <= target_fs:
        return np.asarray(raw_signal), fs_original
    num_samples = int(len(raw_signal) * (target_fs / fs_original))
    return signal.resample(raw_signal, num_samples), target_fs


def apply_mti_filter(raw_signal):
    def mti_pass(sig, beta):
        return signal.lfilter([beta, -beta], [1, -beta], sig)

    x_beta1 = mti_pass(raw_signal, 0.9)
    x_beta2 = mti_pass(raw_signal, 0.99)
    y_filtered = x_beta2 - x_beta1

    y_detrended = signal.detrend(y_filtered)
    y_smoothed = signal.medfilt(y_detrended, kernel_size=5)

    return y_smoothed


def apply_scg_bandpass(raw_signal, fs, low_hz=1.0, high_hz=40.0):
    if fs <= 0:
        return raw_signal
    nyq = 0.5 * fs
    low = max(0.01, float(low_hz)) / nyq
    high = min(0.99 * nyq, float(high_hz)) / nyq
    if low >= high:
        return raw_signal
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, raw_signal)


def svmd(signal_in, max_alpha=2000, tau=0, tol=1e-6, stopc=3, init_omega=0):
    signal_in = np.array(signal_in).flatten()
    if len(signal_in) % 2 > 0:
        signal_in = signal_in[:-1]

    y = savgol_filter(signal_in, window_length=25, polyorder=8)
    signoise = signal_in - y

    save_T = len(signal_in)
    fs = 1.0

    T_half = save_T // 2
    f_mir = np.zeros(2 * save_T)
    f_mir_noise = np.zeros(2 * save_T)

    f_mir[0:T_half] = signal_in[T_half - 1 :: -1]
    f_mir_noise[0:T_half] = signoise[T_half - 1 :: -1]

    f_mir[T_half : save_T + T_half] = signal_in
    f_mir_noise[T_half : save_T + T_half] = signoise

    f_mir[save_T + T_half : 2 * save_T] = signal_in[save_T - 1 : T_half - 1 : -1]
    f_mir_noise[save_T + T_half : 2 * save_T] = signoise[save_T - 1 : T_half - 1 : -1]

    f = f_mir
    fnoise = f_mir_noise
    T = len(f)

    t = np.arange(1, T + 1) / T
    omega_freqs = t - 0.5 - 1.0 / T

    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_onesided = f_hat.copy()
    f_hat_onesided[0 : T // 2] = 0

    f_hat_n = np.fft.fftshift(np.fft.fft(fnoise))
    f_hat_n_onesided = f_hat_n.copy()
    f_hat_n_onesided[0 : T // 2] = 0

    noisepe = np.linalg.norm(f_hat_n_onesided, 2) ** 2

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

    while SC2 != 1:
        while Alpha < (max_alpha + 1):
            udiff = tol + np.finfo(float).eps
            while udiff > tol and n < N_max - 1:
                sum_h_hat = np.sum(h_hat_Temp, axis=0) if len(h_hat_Temp) > 0 else np.zeros(len(omega_freqs))
                sum_u_i = (
                    np.sum(u_hat_i, axis=0)
                    if len(u_hat_i) > 0
                    else np.zeros(len(omega_freqs), dtype=complex)
                )

                freq_diff = omega_freqs - omega_L[n]
                freq_diff_sq = freq_diff**2
                freq_diff_pow4 = freq_diff_sq**2
                alpha_sq = Alpha**2

                num_u = f_hat_onesided + (alpha_sq * freq_diff_pow4) * u_hat_L[n, :] + lambda_arr[n, :] / 2.0
                den_u = 1 + alpha_sq * freq_diff_pow4 * (1 + 2 * Alpha * freq_diff_sq) + sum_h_hat
                u_hat_L[n + 1, :] = num_u / den_u

                pos = slice(T // 2, T)
                mag_sq = np.abs(u_hat_L[n + 1, pos]) ** 2
                den_omega = np.sum(mag_sq)
                if den_omega > 0:
                    omega_L[n + 1] = np.dot(omega_freqs[pos], mag_sq) / den_omega
                else:
                    omega_L[n + 1] = omega_L[n]

                term1 = alpha_sq * freq_diff_pow4
                inner_paren = f_hat_onesided - u_hat_L[n + 1, :] - sum_u_i + lambda_arr[n, :] / 2.0
                num_tau = term1 * inner_paren
                den_tau = 1 + term1
                u_r_updated = num_tau / den_tau + sum_u_i

                lambda_arr[n + 1, :] = lambda_arr[n, :] + tau * (
                    f_hat_onesided - u_hat_L[n + 1, :] - u_r_updated
                )

                udiff = np.finfo(float).eps
                diff_u = u_hat_L[n + 1, :] - u_hat_L[n, :]
                num_udiff = np.sum(diff_u * np.conj(diff_u)).real / T
                den_udiff = np.sum(u_hat_L[n, :] * np.conj(u_hat_L[n, :])).real / T

                if den_udiff > 0:
                    udiff += np.abs(num_udiff / den_udiff)
                udiff = np.abs(udiff)

                n += 1

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

        h_hat_new = gamma[l] / ((alpha_arr[l] ** 2) * (omega_freqs - omega_d_Temp[l]) ** 4)
        h_hat_Temp.append(h_hat_new)
        u_hat_i.append(u_hat_Temp[l])

        sum_u_i = np.sum(u_hat_i, axis=0) if len(u_hat_i) > 0 else np.zeros(len(omega_freqs), dtype=complex)

        if stopc == 1:
            sigerror_val = np.linalg.norm(f_hat_onesided - sum_u_i, 2) ** 2
            sigerror.append(sigerror_val)
            if n2 >= 300 or sigerror[l] <= np.round(noisepe):
                SC2 = 1

        elif stopc == 2:
            sum_u_temp = np.sum(u_hat_Temp, axis=0)
            normind_val = (1 / T) * np.linalg.norm(sum_u_temp - f_hat_onesided, 2) ** 2 / (
                (1 / T) * np.linalg.norm(f_hat_onesided, 2) ** 2
            )
            normind.append(normind_val)
            if n2 >= 300 or normind[l] < 0.005:
                SC2 = 1

        elif stopc == 3:
            sigerror_val = np.linalg.norm(f_hat_onesided - sum_u_i, 2) ** 2
            sigerror.append(sigerror_val)
            BIC_val = 2 * T * np.log(sigerror[l]) + (3 * (l + 1)) * np.log(2 * T)
            BIC.append(BIC_val)
            if l > 0 and BIC[l] > BIC[l - 1]:
                SC2 = 1

        else:
            term = (4 * alpha_arr[l] * u_hat_i[l]) / (1 + 2 * alpha_arr[l] * (omega_freqs - omega_d_Temp[l]) ** 2)
            dot_prod = np.sum(term * np.conj(u_hat_i[l]))
            polm_val = np.abs(dot_prod)

            if l == 0:
                polm.append(polm_val)
                polm_temp = polm[0]
                polm[0] = 1.0
            else:
                polm.append(polm_val / polm_temp)

            if l > 0 and abs(polm[l] - polm[l - 1]) < 0.001:
                SC2 = 1

        u_hat_L = np.zeros((N_max, len(omega_freqs)), dtype=complex)
        omega_L = np.zeros(N_max)
        omega_L[0] = omega_L_start
        n = 0
        l += 1
        m_val = 0
        n2 = 0

    omega_final = np.array(omega_d_Temp)
    L = len(omega_final)
    u_hat = np.zeros((T, L), dtype=complex)

    for idx in range(L):
        u_hat[T // 2 : T, idx] = u_hat_Temp[idx][T // 2 : T]
        u_hat[1 : T // 2, idx] = np.conj(u_hat_Temp[idx][T // 2 + 1 : T][::-1])
        u_hat[0, idx] = np.conj(u_hat[-1, idx])

    u = np.zeros((L, T))
    for idx in range(L):
        u[idx, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, idx])))

    indic = np.argsort(omega_final)
    omega_final = omega_final[indic]
    u = u[indic, :]

    u_final = u[:, T // 4 : 3 * T // 4]

    return u_final, omega_final


def select_ao_modes(modes, omegas=None, fs=None, freq_cutoff_hz=100):
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


def extract_ao_peaks(s_ao, fs, prominence_factor=0.05, power=7):
    taper_window = signal.windows.tukey(len(s_ao), alpha=0.05)
    s_ao_tapered = s_ao * taper_window

    s_ao_7 = np.sign(s_ao_tapered) * np.abs(s_ao_tapered) ** power

    analytic_signal = signal.hilbert(s_ao_7)
    envelope = np.abs(analytic_signal)

    window_width = int(fs / 10)
    smoothed_env = np.convolve(envelope, np.ones(window_width) / window_width, mode="same")

    min_distance = int(0.4 * fs)
    prom_threshold = max(
        np.percentile(smoothed_env, 90) * prominence_factor,
        np.percentile(smoothed_env, 10),
    )
    peaks, _ = signal.find_peaks(smoothed_env, distance=min_distance, prominence=prom_threshold)

    return s_ao_7, envelope, smoothed_env, peaks


def sqa_kurtosis(segment, threshold=7.0):
    return kurtosis(segment) > threshold


def sqa_zcr(segment, fs, low_hz=0.5, high_hz=5.0):
    zcr = np.sum(np.diff(np.sign(segment)) != 0) / (len(segment) / fs)
    return zcr < low_hz or zcr > high_hz


def sqa_flatline(segment, cv_thresh=0.01, diff_thresh=1e-4):
    cv = np.std(segment) / (np.abs(np.mean(segment)) + 1e-9)
    return cv < cv_thresh or np.max(np.abs(np.diff(segment))) < diff_thresh


def sqa_envelope(segment, threshold=2.5):
    env = np.abs(hilbert(segment))
    return (np.std(env) / (np.mean(env) + 1e-9)) > threshold


def _derive_rms_bounds(seg_rms, rms_low_percentile=20, rms_high_percentile=80, rms_low_mad_mult=2.0, rms_high_mad_mult=4.0):
    valid_rms = np.asarray(seg_rms, dtype=float)
    valid_rms = valid_rms[np.isfinite(valid_rms)]
    if len(valid_rms) == 0:
        return 1e-10, float("inf")

    rms_low_perc = float(np.percentile(valid_rms, rms_low_percentile))
    rms_high_perc = float(np.percentile(valid_rms, rms_high_percentile))
    rms_med = float(np.median(valid_rms))
    rms_mad = float(np.median(np.abs(valid_rms - rms_med)))

    rms_low_thr = max(rms_low_perc, rms_med - rms_low_mad_mult * rms_mad) if rms_mad > 1e-12 else rms_low_perc
    rms_high_thr = max(rms_high_perc, rms_med + rms_high_mad_mult * rms_mad) if rms_mad > 1e-12 else rms_high_perc * 3.0
    return float(rms_low_thr), float(rms_high_thr)


def sqa_rms(segment, rms_low_thr, rms_high_thr):
    seg_det = signal.detrend(np.asarray(segment, dtype=float))
    rms_value = float(np.sqrt(np.mean(seg_det ** 2)))
    return (rms_value < rms_low_thr) or (rms_value > rms_high_thr), rms_value


def sqa_combined(
    segment,
    fs,
    min_flags=2,
    kurt_thresh=7.0,
    zcr_low=0.5,
    zcr_high=5.0,
    cv_thresh=0.01,
    diff_thresh=1e-4,
    env_thresh=2.5,
    rms_low_thr=1e-10,
    rms_high_thr=float("inf"),
):
    rms_flag, _ = sqa_rms(segment, rms_low_thr, rms_high_thr)
    flags = [
        sqa_kurtosis(segment, kurt_thresh),
        sqa_zcr(segment, fs, zcr_low, zcr_high),
        sqa_flatline(segment, cv_thresh, diff_thresh),
        sqa_envelope(segment, env_thresh),
        rms_flag,
    ]
    return sum(flags) >= min_flags, flags


def combined_sqa_for_signal(
    signal_in,
    fs,
    segment_seconds=4.0,
    min_flags=2,
    kurt_thresh=7.0,
    zcr_low=0.5,
    zcr_high=5.0,
    cv_thresh=0.01,
    diff_thresh=1e-4,
    env_thresh=2.5,
    rms_low_percentile=20,
    rms_high_percentile=80,
    rms_low_mad_mult=2.0,
    rms_high_mad_mult=4.0,
):
    x = np.asarray(signal_in, dtype=float).flatten()
    n = len(x)
    if n == 0 or fs <= 0:
        return {
            "segment_starts": np.array([]),
            "segment_ends": np.array([]),
            "bad_mask": np.array([], dtype=bool),
            "flags": np.empty((0, 5), dtype=bool),
            "seg_rms": np.array([]),
            "rms_bounds": {"low": np.nan, "high": np.nan},
            "method": "combined",
        }

    seg_len = max(1, int(segment_seconds * fs))
    seg_starts = np.arange(0, n, seg_len)
    seg_ends = np.minimum(seg_starts + seg_len, n)

    seg_rms_all = []
    for s, e in zip(seg_starts, seg_ends):
        seg = x[s:e]
        if len(seg) < 16:
            seg_rms_all.append(np.nan)
            continue
        seg_rms_all.append(float(np.sqrt(np.mean(signal.detrend(seg) ** 2))))

    rms_low_thr, rms_high_thr = _derive_rms_bounds(
        seg_rms_all,
        rms_low_percentile=rms_low_percentile,
        rms_high_percentile=rms_high_percentile,
        rms_low_mad_mult=rms_low_mad_mult,
        rms_high_mad_mult=rms_high_mad_mult,
    )

    bad_mask = []
    flags_all = []
    for s, e, rms_value in zip(seg_starts, seg_ends, seg_rms_all):
        seg = x[s:e]
        if len(seg) < 16:
            bad_mask.append(True)
            flags_all.append([True, True, True, True, True])
            continue
        is_bad, flags = sqa_combined(
            seg,
            fs,
            min_flags=min_flags,
            kurt_thresh=kurt_thresh,
            zcr_low=zcr_low,
            zcr_high=zcr_high,
            cv_thresh=cv_thresh,
            diff_thresh=diff_thresh,
            env_thresh=env_thresh,
            rms_low_thr=rms_low_thr,
            rms_high_thr=rms_high_thr,
        )
        bad_mask.append(is_bad)
        flags_all.append(flags)

    return {
        "segment_starts": seg_starts,
        "segment_ends": seg_ends,
        "bad_mask": np.asarray(bad_mask, dtype=bool),
        "flags": np.asarray(flags_all, dtype=bool),
        "seg_rms": np.asarray(seg_rms_all, dtype=float),
        "rms_bounds": {"low": float(rms_low_thr), "high": float(rms_high_thr)},
        "method": "combined",
    }


def build_sample_bad_mask(signal_len, sqa_result):
    sample_bad = np.zeros(signal_len, dtype=bool)
    seg_starts = sqa_result.get("segment_starts", [])
    seg_ends = sqa_result.get("segment_ends", [])
    bad_mask = sqa_result.get("bad_mask", [])

    for s, e, is_bad in zip(seg_starts, seg_ends, bad_mask):
        sample_bad[int(s) : int(e)] = bool(is_bad)

    return sample_bad


def _iqr_inlier_mask(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=bool)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return np.ones(values.shape, dtype=bool)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (values >= lower) & (values <= upper)


def _estimate_peak_lag_samples(detected_peaks, reference_peaks, fs_ao, fs_ref):
    detected = np.asarray(detected_peaks, dtype=float)
    reference = np.asarray(reference_peaks, dtype=float)
    if detected.size == 0 or reference.size == 0:
        return 0.0

    detected_s = detected / float(fs_ao)
    reference_s = reference / float(fs_ref)
    min_lag_s = 0.05
    max_lag_s = 0.50
    if max_lag_s <= min_lag_s:
        return 0.0

    diffs = []
    for ref in reference_s:
        lo = ref - max_lag_s
        hi = ref - min_lag_s
        nearby = detected_s[(detected_s >= lo) & (detected_s <= hi)]
        if nearby.size:
            diffs.extend(nearby - ref)

    if len(diffs) == 0:
        return 0.0

    diffs = np.asarray(diffs, dtype=float)
    bin_width = 0.001
    diffs_int = np.round(diffs / bin_width).astype(int)
    offset = -int(diffs_int.min())
    counts = np.bincount(diffs_int + offset)
    best_lag = (int(np.argmax(counts)) - offset) * bin_width
    return float(best_lag)


def _ptt_window_samples(lag_seconds, tolerance_seconds=0.15):
    window_min = lag_seconds - float(tolerance_seconds)
    window_max = lag_seconds + float(tolerance_seconds)
    if window_min >= window_max:
        window_min = lag_seconds - float(tolerance_seconds)
        window_max = lag_seconds + float(tolerance_seconds)
    return window_min, window_max


def _match_peaks_by_lag(detected_peaks, reference_peaks, fs_ao, fs_ref, tolerance_seconds=0.15):
    detected = np.asarray(detected_peaks, dtype=float)
    reference = np.asarray(reference_peaks, dtype=float)
    if detected.size == 0 or reference.size == 0:
        return [], []

    detected = np.sort(detected)
    reference = np.sort(reference)
    detected_s = detected / float(fs_ao)
    reference_s = reference / float(fs_ref)
    lag_seconds = _estimate_peak_lag_samples(detected, reference, fs_ao, fs_ref)
    window_min, window_max = _ptt_window_samples(lag_seconds, tolerance_seconds)

    used = np.zeros(len(detected_s), dtype=bool)
    matched_detected = []
    matched_reference = []

    for ref_idx, ref in enumerate(reference_s):
        candidates = np.where(
            (~used)
            & (detected_s - ref >= window_min)
            & (detected_s - ref <= window_max)
        )[0]
        if len(candidates) == 0:
            continue
        closest_idx = candidates[np.argmin(np.abs(detected_s[candidates] - ref))]
        used[closest_idx] = True
        matched_detected.append(int(closest_idx))
        matched_reference.append(int(ref_idx))

    return matched_detected, matched_reference


def compute_ptt_metrics(ao_peaks, ppg_peaks, fs_ao, fs_ref, tolerance_seconds=0.15):
    ao_peaks = np.asarray(ao_peaks, dtype=float)
    ppg_peaks = np.asarray(ppg_peaks, dtype=float)
    if ao_peaks.size == 0 or ppg_peaks.size == 0:
        return None

    ao_sorted = np.sort(ao_peaks)
    ppg_sorted = np.sort(ppg_peaks)
    ao_s = ao_sorted / float(fs_ao)
    ppg_s = ppg_sorted / float(fs_ref)
    matched_ao_idx, matched_ppg_idx = _match_peaks_by_lag(
        ao_sorted, ppg_sorted, fs_ao, fs_ref, tolerance_seconds
    )
    if len(matched_ao_idx) == 0:
        return None

    ptt_ms = (ppg_s[np.asarray(matched_ppg_idx)] - ao_s[np.asarray(matched_ao_idx)]) * 1000.0

    ao_intervals_ms = np.diff(ao_s) * 1000.0
    ao_interval_series = []
    ptt_series = []
    for ao_idx, ptt_value in zip(matched_ao_idx, ptt_ms):
        if ao_idx == 0:
            continue
        ao_interval_series.append(ao_intervals_ms[ao_idx - 1])
        ptt_series.append(ptt_value)

    ptt_corr = (
        np.corrcoef(ptt_series, ao_interval_series)[0, 1]
        if len(ptt_series) > 1
        else np.nan
    )

    return {
        "mean_ptt_ms": float(np.mean(ptt_ms)),
        "std_ptt_ms": float(np.std(ptt_ms)),
        "ptt_rr_correlation": ptt_corr,
    }


def match_intervals_by_time(ao_peaks, ref_peaks, fs, apply_iqr=False):
    ao_peaks = np.asarray(ao_peaks, dtype=float)
    ref_peaks = np.asarray(ref_peaks, dtype=float)

    if len(ao_peaks) < 2 or len(ref_peaks) < 2:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    ao_intervals_ms = np.diff(ao_peaks) / fs * 1000.0
    ref_intervals_ms = np.diff(ref_peaks) / fs * 1000.0
    ao_centers_s = (ao_peaks[:-1] + ao_peaks[1:]) / 2.0 / fs
    ref_centers_s = (ref_peaks[:-1] + ref_peaks[1:]) / 2.0 / fs

    if len(ao_centers_s) == 0 or len(ref_centers_s) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    median_ref_seconds = np.median(ref_intervals_ms) / 1000.0
    if not np.isfinite(median_ref_seconds) or median_ref_seconds <= 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    max_time_diff = 0.5 * median_ref_seconds

    matched_ao = []
    matched_ref = []
    matched_ao_centers = []
    matched_ref_centers = []

    for idx, ao_center in enumerate(ao_centers_s):
        nearest_idx = int(np.argmin(np.abs(ref_centers_s - ao_center)))
        time_diff = abs(ref_centers_s[nearest_idx] - ao_center)
        if time_diff < max_time_diff:
            matched_ao.append(ao_intervals_ms[idx])
            matched_ref.append(ref_intervals_ms[nearest_idx])
            matched_ao_centers.append(ao_center)
            matched_ref_centers.append(ref_centers_s[nearest_idx])

    matched_ao = np.asarray(matched_ao, dtype=float)
    matched_ref = np.asarray(matched_ref, dtype=float)
    matched_ao_centers = np.asarray(matched_ao_centers, dtype=float)
    matched_ref_centers = np.asarray(matched_ref_centers, dtype=float)
    if matched_ao.size == 0 or matched_ref.size == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    removed_ao_centers = np.array([])
    removed_ref_centers = np.array([])
    if apply_iqr:
        ao_mask = _iqr_inlier_mask(matched_ao)
        ref_mask = _iqr_inlier_mask(matched_ref)
        keep_mask = ao_mask & ref_mask
        removed_ao_centers = matched_ao_centers[~keep_mask]
        removed_ref_centers = matched_ref_centers[~keep_mask]
    else:
        keep_mask = np.ones(matched_ao.shape, dtype=bool)

    return (
        matched_ao[keep_mask],
        matched_ref[keep_mask],
        matched_ao_centers[keep_mask],
        matched_ref_centers[keep_mask],
        removed_ao_centers,
        removed_ref_centers,
    )


def compute_paper_metrics(ao_intervals_ms, ref_intervals_ms):
    ao_intervals_ms = np.asarray(ao_intervals_ms, dtype=float)
    ref_intervals_ms = np.asarray(ref_intervals_ms, dtype=float)
    if len(ao_intervals_ms) == 0 or len(ref_intervals_ms) == 0:
        return None

    ao_hr = 60000.0 / ao_intervals_ms
    ref_hr = 60000.0 / ref_intervals_ms

    mean_scg_hr = float(np.mean(ao_hr))
    mean_ref_hr = float(np.mean(ref_hr))
    are = abs(mean_scg_hr - mean_ref_hr) / mean_ref_hr if mean_ref_hr != 0 else np.nan
    aae = float(np.mean(np.abs(ao_hr - ref_hr)))
    aaep = float(np.mean(np.abs(ao_hr - ref_hr) / ref_hr) * 100.0)

    diff_intervals = ao_intervals_ms - ref_intervals_ms
    bias = float(np.mean(diff_intervals))
    std_diff = float(np.std(diff_intervals))
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff

    return {
        "ARE": are,
        "AAE": aae,
        "AAEP": aaep,
        "mean_scg_hr": mean_scg_hr,
        "mean_ref_hr": mean_ref_hr,
        "ba_bias": bias,
        "ba_upper_loa": loa_upper,
        "ba_lower_loa": loa_lower,
    }


def compute_detection_metrics(detected_peaks, reference_peaks, fs_ao, fs_ref, tolerance_seconds=0.15):
    detected = np.asarray(detected_peaks, dtype=float)
    reference = np.asarray(reference_peaks, dtype=float)
    if len(reference) == 0:
        return None

    detected = np.sort(detected)
    reference = np.sort(reference)
    detected_s = detected / float(fs_ao)
    reference_s = reference / float(fs_ref)
    lag_seconds = _estimate_peak_lag_samples(detected, reference, fs_ao, fs_ref)
    window_min, window_max = _ptt_window_samples(lag_seconds, tolerance_seconds)

    used = np.zeros(len(detected_s), dtype=bool)
    matched_ref = np.zeros(len(reference_s), dtype=bool)
    tp = 0

    for ref_idx, ref in enumerate(reference_s):
        candidates = np.where(
            (~used)
            & (detected_s - ref >= window_min)
            & (detected_s - ref <= window_max)
        )[0]
        if len(candidates) == 0:
            continue
        closest_idx = candidates[np.argmin(np.abs(detected_s[candidates] - ref))]
        used[closest_idx] = True
        matched_ref[ref_idx] = True
        tp += 1

    fp = int(np.sum(~used))
    fn = int(np.sum(~matched_ref))
    fp_peaks = detected[~used]
    fn_peaks = reference[~matched_ref]

    se = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else np.nan
    p = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else np.nan
    acc = tp / (tp + fp + fn) * 100.0 if (tp + fp + fn) > 0 else np.nan
    der = (fp + fn) / (tp + fn) * 100.0 if (tp + fn) > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SE": se,
        "P": p,
        "ACC": acc,
        "DER": der,
        "fp_peaks": fp_peaks,
        "fn_peaks": fn_peaks,
        "fp_times_s": fp_peaks / float(fs_ao),
        "fn_times_s": fn_peaks / float(fs_ref),
    }


with st.sidebar:
    st.header("Data Source")
    source_mode = st.radio("Choose source", ["Workspace CSV", "Upload CSV"], index=0)

    uploaded = None
    selected_file_path = ""

    if source_mode == "Workspace CSV":
        search_dir = st.text_input("Search folder", value=DEFAULT_SEARCH_DIR)
        path_obj = Path(search_dir)
        if path_obj.exists() and path_obj.is_dir():
            date_folders = sorted([p for p in path_obj.iterdir() if p.is_dir()])
            if date_folders:
                selected_day = st.selectbox("Day folder", [p.name for p in date_folders])
                day_path = path_obj / selected_day
                candidates = sorted(day_path.glob("*.csv"))
            else:
                candidates = sorted(path_obj.glob("*.csv"))

            if candidates:
                selected = st.selectbox("CSV file", [str(p) for p in candidates])
                selected_file_path = selected
            else:
                st.warning("No CSV files found in that folder.")
        else:
            st.warning("Folder does not exist.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    expected_hz = st.number_input("Expected sample rate (Hz)", min_value=1.0, value=256.0, step=1.0)
    override_fs = st.checkbox("Override inferred sample rate", value=False)
    override_hz = st.number_input("Override sample rate (Hz)", min_value=1.0, value=256.0, step=1.0)

    st.divider()
    st.subheader("Preprocessing")
    preprocessing_mode = st.selectbox(
        "Preprocessing",
        ["MTI + Detrend + Median", "Bandpass 1-40 Hz", "None"],
        index=0,
    )
    target_fs = st.number_input("Processing sample rate (Hz)", min_value=50, max_value=1000, value=256, step=50)

    st.divider()
    st.subheader("SQA: Signal Quality Assessment")
    show_sqa_overlay = st.checkbox(
        "Enable SQA (Highlight low-quality SCG segments)",
        value=True,
        help="Applies only to Full Record Analysis. Bad segments are shown in red.",
    )
    sqa_segment_seconds = st.slider("SQA Segment Length (s)", 3, 5, 4, 1)
    min_flags_to_reject = st.slider(
        "Minimum detectors to flag a window as bad",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
    )

    with st.expander("Advanced detector thresholds"):
        kurt_thresh = st.slider("Kurtosis threshold", 3.0, 20.0, 7.0, 0.5)
        zcr_low = st.slider("ZCR low bound (Hz)", 0.1, 2.0, 0.5, 0.1)
        zcr_high = st.slider("ZCR high bound (Hz)", 2.0, 20.0, 5.0, 0.5)
        env_thresh = st.slider("Envelope CV threshold", 1.0, 5.0, 2.5, 0.1)
        col_rms1, col_rms2 = st.columns(2)
        with col_rms1:
            rms_low_percentile = st.slider("RMS low percentile", 5, 30, 20, 1)
            rms_low_mad_mult = st.slider("RMS low MAD multiplier", 0.5, 4.0, 2.0, 0.25)
        with col_rms2:
            rms_high_percentile = st.slider("RMS high percentile", 70, 95, 80, 1)
            rms_high_mad_mult = st.slider("RMS high MAD multiplier", 2.0, 8.0, 4.0, 0.25)

    exclude_bad_windows = st.checkbox(
        "Skip very noisy windows in Full Record Analysis",
        value=True,
    )
    show_sqa_breakdown = st.checkbox("Show SQA breakdown", value=False)
    bad_window_fraction_threshold = st.slider(
        "Skip window if bad fraction >=",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
    )

    st.divider()
    st.subheader("SVMD + Peak Detection")
    svmd_alpha = st.slider("SVMD Alpha (Bandwidth)", 100, 2000, 260)
    prominence_factor = st.slider(
        "Peak Prominence Threshold",
        0.01,
        0.30,
        0.05,
        0.01,
    )
    power_exp = st.slider("Power-Law Exponent", 3, 9, 7)

    st.divider()
    st.subheader("PPG Beat Source")
    beat_source = st.radio(
        "Beat reference",
        ["Use CSV beat_event", "Detect from PPG raw"],
        index=0,
        help="If PPG raw exists, you can detect systolic peaks via bandpass + find_peaks.",
    )
    ppg_bp_low = st.number_input("PPG bandpass low (Hz)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    ppg_bp_high = st.number_input("PPG bandpass high (Hz)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
    ppg_max_bpm = st.number_input("Max BPM for peak distance", min_value=60.0, max_value=300.0, value=200.0, step=10.0)
    ppg_prom = st.number_input("PPG peak prominence factor", min_value=0.0, value=0.05, step=0.01)

    st.divider()
    st.subheader("Interval Metrics")
    use_iqr_filter = st.checkbox(
        "Apply IQR outlier filtering",
        value=True,
        help="Removes intervals outside 1.5x IQR for both AO and PPG series.",
    )

    st.divider()
    st.subheader("Outputs")
    save_json_output = st.checkbox("Save AO Peaks to JSON", value=False)
    save_ppg_json_output = st.checkbox(
        "Save PPG Peaks to JSON",
        value=False,
        disabled=beat_source != "Detect from PPG raw",
        help="Available when raw PPG peaks are detected from the PPG signal.",
    )
    output_folder = st.text_input("Output Folder", value="Saved_Peaks")


if uploaded is None and not selected_file_path:
    st.info("Select or upload a CSV file to begin.")
    st.stop()

meta_data = None
meta_path_display = ""
if uploaded is None and selected_file_path:
    try:
        meta_data, meta_path_display = _read_metadata_for_csv(selected_file_path)
    except Exception as exc:
        st.warning(f"Failed to read metadata: {exc}")

try:
    raw_df = _read_csv_from_source(uploaded, selected_file_path)
    df = _prepare_df(raw_df)
except Exception as exc:
    st.error(f"Failed to load CSV: {exc}")
    st.stop()

if meta_data is not None:
    st.subheader("Session Metadata")
    st.json(meta_data)
    if meta_path_display:
        st.caption(f"Metadata file: {meta_path_display}")

scg_df = df[df[["x_g", "y_g", "z_g"]].notna().any(axis=1)].copy()
beats_df = df[df["beat_event"] == 1].copy()
ppg_df = df[df[OPTIONAL_PPG_COL].notna()].copy() if OPTIONAL_PPG_COL in df.columns else pd.DataFrame()

if scg_df.empty:
    st.error("No SCG rows found (x_g/y_g/z_g are empty for all rows).")
    st.stop()

file_label = Path(selected_file_path).stem if selected_file_path else "uploaded"

duration_s, actual_hz = _compute_rate(scg_df)
fs_infer = override_hz if override_fs else actual_hz

diff_hz = actual_hz - expected_hz
pct = (diff_hz / expected_hz * 100.0) if expected_hz > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("SCG Samples", f"{len(scg_df):,}")
m2.metric("PPG Beats", f"{len(beats_df):,}")
m3.metric("Actual Rate", f"{actual_hz:.2f} Hz")
m4.metric("Rate Error", f"{diff_hz:+.2f} Hz", f"{pct:+.2f}%")

st.caption(f"Capture duration: {duration_s:.2f} s")

if fs_infer <= 0:
    st.error("Invalid sampling rate inferred. Check timestamp_ms.")
    st.stop()

scg_df = scg_df.copy()
beats_df = beats_df.copy()

t0_ms = float(scg_df["timestamp_ms"].iloc[0])
scg_df["time_s"] = (scg_df["timestamp_ms"] - t0_ms) / 1000.0
beats_df["time_s"] = (beats_df["timestamp_ms"] - t0_ms) / 1000.0 if not beats_df.empty else np.array([])
if not ppg_df.empty:
    ppg_df["time_s"] = (ppg_df["timestamp_ms"] - t0_ms) / 1000.0

max_t = float(scg_df["time_s"].iloc[-1])

st.divider()
st.subheader("Window Analysis")

window_col1, window_col2 = st.columns([2, 1])
with window_col1:
    if max_t >= 1.0:
        max_window = min(30.0, max_t)
        window_size = st.slider("Window Size (s)", 1.0, max_window, min(10.0, max_window), 0.5)
        start_max = max(0.0, max_t - window_size)
        start_time = st.slider("Start Time (s)", 0.0, start_max, 0.0, 0.1)
    else:
        window_size = 0.0
        start_time = 0.0
with window_col2:
    run_window_btn = st.button("Run SVMD on Window")

scg_raw = scg_df["z_g"].to_numpy(dtype=float)

scg_proc_full, fs_proc = resample_for_processing(scg_raw, fs_infer, target_fs=target_fs)
if preprocessing_mode == "MTI + Detrend + Median":
    scg_proc_full = apply_mti_filter(scg_proc_full)
elif preprocessing_mode == "Bandpass 1-40 Hz":
    scg_proc_full = apply_scg_bandpass(scg_proc_full, fs_proc, low_hz=1.0, high_hz=40.0)

beat_times_s = beats_df["time_s"].to_numpy(dtype=float) if len(beats_df) > 0 else np.array([])

ppg_filtered = None
ppg_peaks_idx = np.array([], dtype=int)
ppg_fs = 0.0
ppg_vis_filtered = None
ppg_vis_peaks_idx = np.array([], dtype=int)
ppg_vis_fs = 0.0

if not ppg_df.empty:
    ppg_ts = ppg_df["timestamp_ms"].to_numpy(dtype=float)
    _, ppg_vis_fs = _compute_rate_from_ts(ppg_ts)
    ppg_signal = ppg_df[OPTIONAL_PPG_COL].to_numpy(dtype=float)
    ppg_vis_filtered = _bandpass_ppg(
        ppg_signal,
        ppg_vis_fs,
        float(ppg_bp_low),
        float(ppg_bp_high),
    )
    ppg_vis_peaks_idx = _detect_ppg_peaks(
        ppg_vis_filtered,
        ppg_vis_fs,
        float(ppg_max_bpm),
        float(ppg_prom),
    )

if beat_source == "Detect from PPG raw" and not ppg_df.empty:
    ppg_filtered = ppg_vis_filtered
    ppg_fs = ppg_vis_fs
    ppg_peaks_idx = ppg_vis_peaks_idx
    if len(ppg_peaks_idx) > 0:
        beat_times_s = ppg_df["time_s"].to_numpy(dtype=float)[ppg_peaks_idx]

ref_fs = float(ppg_vis_fs) if beat_source == "Detect from PPG raw" and ppg_vis_fs > 0 else 100.0
ppg_peaks_ref = (beat_times_s * ref_fs).astype(int) if len(beat_times_s) > 0 else np.array([], dtype=int)

ppg_peaks_full = (beat_times_s * fs_proc).astype(int)
ppg_peaks_full = ppg_peaks_full[(ppg_peaks_full >= 0) & (ppg_peaks_full < len(scg_proc_full))]

if ppg_vis_filtered is not None:
    st.subheader("Processed PPG and Detected Peaks")
    ppg_time_s = ppg_df["time_s"].to_numpy(dtype=float)
    fig_ppg = go.Figure()
    fig_ppg.add_trace(
        go.Scatter(
            x=ppg_time_s,
            y=ppg_vis_filtered,
            mode="lines",
            name="PPG (Bandpassed)",
            line=dict(width=1.2, color="#ff4757"),
        )
    )
    if beat_source == "Detect from PPG raw":
        plot_peaks_idx = ppg_vis_peaks_idx
    else:
        plot_peaks_idx = _map_times_to_indices(ppg_time_s, beat_times_s)
    if len(plot_peaks_idx) > 0:
        fig_ppg.add_trace(
            go.Scatter(
                x=ppg_time_s[plot_peaks_idx],
                y=ppg_vis_filtered[plot_peaks_idx],
                mode="markers",
                name="PPG Peaks",
                marker=dict(color="#00e5ff", size=6, symbol="circle"),
            )
        )
    fig_ppg.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="PPG (Filtered)",
        height=320,
        plot_bgcolor="white",
        showlegend=True,
    )
    st.plotly_chart(fig_ppg, width="stretch")

if run_window_btn:
    start_idx = int(start_time * fs_proc)
    end_idx = int((start_time + window_size) * fs_proc)
    end_idx = min(end_idx, len(scg_proc_full))

    scg_window = scg_proc_full[start_idx:end_idx]

    with st.spinner("Running SVMD on window..."):
        modes, omegas = svmd(scg_window, max_alpha=svmd_alpha, tau=0, stopc=3)

    if len(omegas) == 0:
        st.warning("SVMD returned no modes. Try adjusting parameters.")
    else:
        s_ao, wfs, wf_mean, selected_idx = select_ao_modes(modes, omegas, fs_proc)
        s_ao_7, envelope, smoothed_env, peaks = extract_ao_peaks(
            s_ao, fs_proc, prominence_factor, power=power_exp
        )

        ppg_peaks_window = ppg_peaks_full[(ppg_peaks_full >= start_idx) & (ppg_peaks_full < end_idx)]
        ppg_peaks_window = ppg_peaks_window - start_idx
        ppg_window_mask = (beat_times_s >= start_time) & (beat_times_s < start_time + window_size)
        ppg_peaks_window_ref = ppg_peaks_ref[ppg_window_mask]
        peaks_abs = peaks + start_idx

        comparison = None
        detection_metrics_window = None
        iqr_removed_ao_centers_s = np.array([])
        iqr_removed_ppg_centers_s = np.array([])
        if len(peaks) > 0 and len(ppg_peaks_window_ref) > 0:
            detection_metrics_window = compute_detection_metrics(
                peaks_abs,
                ppg_peaks_window_ref,
                fs_proc,
                ref_fs,
                tolerance_seconds=0.2,
            )
        if len(peaks) > 1 and len(ppg_peaks_window) > 1:
            (
                ao_intervals_ms,
                ppg_intervals_ms,
                ao_centers_s,
                ppg_centers_s,
                iqr_removed_ao_centers_s,
                iqr_removed_ppg_centers_s,
            ) = match_intervals_by_time(
                peaks,
                ppg_peaks_window,
                fs_proc,
                apply_iqr=use_iqr_filter,
            )
            if len(ao_intervals_ms) > 0:
                correlation = (
                    np.corrcoef(ao_intervals_ms, ppg_intervals_ms)[0, 1]
                    if len(ao_intervals_ms) > 1
                    else np.nan
                )
                rmse = np.sqrt(np.mean((ao_intervals_ms - ppg_intervals_ms) ** 2))
                mae = np.mean(np.abs(ao_intervals_ms - ppg_intervals_ms))
                mean_intervals = (ao_intervals_ms + ppg_intervals_ms) / 2
                diff_intervals = ao_intervals_ms - ppg_intervals_ms
                mean_diff = np.mean(diff_intervals)
                std_diff = np.std(diff_intervals)
                paper_metrics = compute_paper_metrics(ao_intervals_ms, ppg_intervals_ms)
                ptt_metrics = compute_ptt_metrics(peaks_abs, ppg_peaks_window_ref, fs_proc, ref_fs)

                comparison = {
                    "ao_intervals_ms": ao_intervals_ms,
                    "ppg_intervals_ms": ppg_intervals_ms,
                    "ao_centers_s": ao_centers_s,
                    "ppg_centers_s": ppg_centers_s,
                    "iqr_removed_ao_centers_s": iqr_removed_ao_centers_s,
                    "iqr_removed_ppg_centers_s": iqr_removed_ppg_centers_s,
                    "correlation": correlation,
                    "rmse": rmse,
                    "mae": mae,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "mean_intervals": mean_intervals,
                    "diff_intervals": diff_intervals,
                    "paper_metrics": paper_metrics,
                    "ptt_metrics": ptt_metrics,
                }

        time_axis = np.linspace(start_time, start_time + window_size, len(scg_window))

        fig_overlay = go.Figure()
        fig_overlay.add_trace(
            go.Scatter(x=time_axis, y=scg_window, mode="lines", name="Processed SCG")
        )
        if len(peaks) > 0:
            fig_overlay.add_trace(
                go.Scatter(
                    x=time_axis[peaks],
                    y=scg_window[peaks],
                    mode="markers",
                    name="AO Peaks",
                    marker=dict(color="red", size=8),
                )
            )
        if len(ppg_peaks_window) > 0:
            for bt in time_axis[ppg_peaks_window]:
                fig_overlay.add_vline(x=float(bt), line_width=1, line_dash="dash", line_color="green")
        if use_iqr_filter:
            for center_s in iqr_removed_ao_centers_s:
                fig_overlay.add_vline(
                    x=float(start_time + center_s),
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(255, 140, 0, 0.7)",
                )
            for center_s in iqr_removed_ppg_centers_s:
                fig_overlay.add_vline(
                    x=float(start_time + center_s),
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(0, 229, 255, 0.7)",
                )

        if detection_metrics_window:
            fp_peaks = detection_metrics_window.get("fp_peaks", np.array([], dtype=int)).astype(int)
            fn_times_s = detection_metrics_window.get("fn_times_s", np.array([], dtype=float))
            fp_peaks = fp_peaks - start_idx
            fp_peaks = fp_peaks[(fp_peaks >= 0) & (fp_peaks < len(scg_window))]
            if len(fp_peaks) > 0:
                fig_overlay.add_trace(
                    go.Scatter(
                        x=time_axis[fp_peaks],
                        y=scg_window[fp_peaks],
                        mode="markers",
                        name="False Positives",
                        marker=dict(color="#ff8c00", size=9, symbol="triangle-up"),
                    )
                )
            if len(fn_times_s) > 0:
                for bt in fn_times_s:
                    fig_overlay.add_vline(x=float(bt), line_width=1, line_dash="dot", line_color="#6a5acd")

        fig_overlay.update_layout(
            title="Processed SCG with AO Peaks and PPG Beats",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=360,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_overlay, width="stretch")

        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(x=time_axis, y=s_ao, mode="lines", name="Reconstructed AO"))
        fig_env.add_trace(go.Scatter(x=time_axis, y=envelope, mode="lines", name="Envelope"))
        fig_env.add_trace(go.Scatter(x=time_axis, y=smoothed_env, mode="lines", name="Smoothed Env"))
        if len(peaks) > 0:
            fig_env.add_trace(
                go.Scatter(
                    x=time_axis[peaks],
                    y=smoothed_env[peaks],
                    mode="markers",
                    name="Detected AO Peaks",
                    marker=dict(color="red", size=6),
                )
            )
        if use_iqr_filter:
            for center_s in iqr_removed_ao_centers_s:
                fig_env.add_vline(
                    x=float(start_time + center_s),
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(255, 140, 0, 0.7)",
                )
            for center_s in iqr_removed_ppg_centers_s:
                fig_env.add_vline(
                    x=float(start_time + center_s),
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(0, 229, 255, 0.7)",
                )
        fig_env.update_layout(
            title="AO Reconstruction and Envelope",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=360,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_env, width="stretch")

        if comparison is not None:
            st.subheader("AO-AO vs PPG-PPG Interval Comparison")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Correlation", f"{comparison['correlation']:.3f}")
            col2.metric("RMSE", f"{comparison['rmse']:.1f} ms")
            col3.metric("MAE", f"{comparison['mae']:.1f} ms")
            col4.metric("Bias", f"{comparison['mean_diff']:.1f} ms")

            paper_metrics = comparison.get("paper_metrics")
            if paper_metrics:
                col5, col6, col7, col8, col9 = st.columns(5)
                col5.metric("Mean SCG HR (BPM)", f"{paper_metrics['mean_scg_hr']:.1f}")
                col6.metric("Mean PPG HR (BPM)", f"{paper_metrics['mean_ref_hr']:.1f}")
                col7.metric("ARE", f"{paper_metrics['ARE']:.3f}")
                col8.metric("AAE (BPM)", f"{paper_metrics['AAE']:.2f}")
                col9.metric("AAEP (%)", f"{paper_metrics['AAEP']:.2f}")

            if use_iqr_filter:
                removed_ao = comparison.get("iqr_removed_ao_centers_s", np.array([]))
                removed_ppg = comparison.get("iqr_removed_ppg_centers_s", np.array([]))
                if removed_ao.size > 0 or removed_ppg.size > 0:
                    with st.expander("IQR-filtered intervals", expanded=False):
                        st.write(
                            f"Removed AO intervals: {len(removed_ao)} | "
                            f"Removed PPG intervals: {len(removed_ppg)}"
                        )
                        if removed_ao.size > 0:
                            st.dataframe(
                                pd.DataFrame({"ao_interval_center_s": removed_ao + start_time}),
                                width="stretch",
                                hide_index=True,
                            )
                        if removed_ppg.size > 0:
                            st.dataframe(
                                pd.DataFrame({"ppg_interval_center_s": removed_ppg + start_time}),
                                width="stretch",
                                hide_index=True,
                            )

            detection_metrics = detection_metrics_window
            if detection_metrics:
                col10, col11, col12, col13, col14, col15, col16 = st.columns(7)
                col10.metric("TP", f"{detection_metrics['TP']}")
                col11.metric("FP", f"{detection_metrics['FP']}")
                col12.metric("FN", f"{detection_metrics['FN']}")
                col13.metric("SE (%)", f"{detection_metrics['SE']:.1f}")
                col14.metric("P (%)", f"{detection_metrics['P']:.1f}")
                col15.metric("ACC (%)", f"{detection_metrics['ACC']:.1f}")
                col16.metric("DER (%)", f"{detection_metrics['DER']:.1f}")

                with st.expander("Detection errors", expanded=False):
                    fp_times = detection_metrics.get("fp_times_s", np.array([]))
                    fn_times = detection_metrics.get("fn_times_s", np.array([]))
                    fp_peaks = detection_metrics.get("fp_peaks", np.array([]))
                    fn_peaks = detection_metrics.get("fn_peaks", np.array([]))
                    st.write(f"False positives: {len(fp_times)} | False negatives: {len(fn_times)}")
                    if len(fp_times) > 0:
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "fp_sample_idx": fp_peaks,
                                    "fp_time_s": fp_times,
                                }
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    if len(fn_times) > 0:
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "fn_sample_idx": fn_peaks,
                                    "fn_time_s": fn_times,
                                }
                            ),
                            width="stretch",
                            hide_index=True,
                        )


st.divider()
st.subheader("Full Record Analysis")

col_trim1, col_trim2 = st.columns(2)
with col_trim1:
    trim_start = st.number_input("Trim Start (s)", min_value=0.0, max_value=max_t, value=0.0, step=1.0)
with col_trim2:
    trim_end = st.number_input("Trim End (s)", min_value=0.0, max_value=max_t, value=max_t, step=1.0)

if trim_start >= trim_end:
    st.error("Trim Start must be strictly less than Trim End.")

full_record_btn = st.button("Analyze Full Record", disabled=trim_start >= trim_end)

if full_record_btn:
    start_idx_global = int(trim_start * fs_proc)
    end_idx_global = int(trim_end * fs_proc)
    
    scg_proc_full = scg_proc_full[start_idx_global:end_idx_global]
    scg_raw = scg_raw[int(trim_start * fs_infer):int(trim_end * fs_infer)]
    
    ppg_peaks_full = ppg_peaks_full[(ppg_peaks_full >= start_idx_global) & (ppg_peaks_full < end_idx_global)]
    ppg_peaks_full = ppg_peaks_full - start_idx_global
    ppg_ref_mask = (beat_times_s >= trim_start) & (beat_times_s < trim_end)
    ppg_peaks_ref_trim = ppg_peaks_ref[ppg_ref_mask]
    
    analysis_duration = trim_end - trim_start
    st.info(f"Analyzing trimmed record ({analysis_duration:.1f}s) using 10-second windows...")

    all_ao_peaks = []
    all_ao_intervals = []
    all_ao_intervals_times = []

    window_duration = 10.0
    total_duration = len(scg_proc_full) / fs_proc
    num_windows = int(np.ceil(total_duration / window_duration))

    sqa_result_full_record = None
    sample_bad_mask_full_record = None

    if show_sqa_overlay:
        sqa_result_full_record = combined_sqa_for_signal(
            scg_proc_full,
            fs=fs_proc,
            segment_seconds=float(sqa_segment_seconds),
            min_flags=int(min_flags_to_reject),
            kurt_thresh=float(kurt_thresh),
            zcr_low=float(zcr_low),
            zcr_high=float(zcr_high),
            env_thresh=float(env_thresh),
            rms_low_percentile=int(rms_low_percentile),
            rms_high_percentile=int(rms_high_percentile),
            rms_low_mad_mult=float(rms_low_mad_mult),
            rms_high_mad_mult=float(rms_high_mad_mult),
        )
        sample_bad_mask_full_record = build_sample_bad_mask(len(scg_proc_full), sqa_result_full_record)

        bad_pct_full = float(np.mean(sample_bad_mask_full_record) * 100.0)
        st.caption(
            f"Combined SQA | Min flags: {min_flags_to_reject} | Bad samples: {bad_pct_full:.1f}%"
        )

        if show_sqa_breakdown:
            flag_rows = sqa_result_full_record.get("flags", np.empty((0, 5), dtype=bool))
            if len(flag_rows) > 0:
                sqa_breakdown_df = pd.DataFrame(
                    flag_rows,
                    columns=["Kurtosis", "ZCR", "Flatline", "Envelope", "RMS"],
                )
                st.dataframe(sqa_breakdown_df, width="stretch", hide_index=True)

    status_text = st.empty()
    progress_bar = st.progress(0)
    skipped_bad_windows = 0

    wall_clock_start = time.time()

    for i in range(num_windows):
        window_start = i * window_duration
        window_end = min(window_start + window_duration, total_duration)

        status_text.text(
            f"Processing window {i + 1}/{num_windows} ({window_start:.1f}s - {window_end:.1f}s)"
        )
        progress_bar.progress(int((i + 1) / max(1, num_windows) * 100))

        start_idx_w = int(window_start * fs_proc)
        end_idx_w = int(window_end * fs_proc)
        if end_idx_w <= start_idx_w:
            continue

        if sample_bad_mask_full_record is not None and exclude_bad_windows:
            bad_frac_window = float(np.mean(sample_bad_mask_full_record[start_idx_w:end_idx_w]))
            if bad_frac_window >= float(bad_window_fraction_threshold):
                skipped_bad_windows += 1
                continue

        scg_window = scg_proc_full[start_idx_w:end_idx_w]

        modes_w, omegas_w = svmd(scg_window, max_alpha=svmd_alpha, tau=0, stopc=3)
        if len(omegas_w) == 0:
            continue

        s_ao_w, _, _, _ = select_ao_modes(modes_w, omegas_w, fs_proc)
        _, _, _, ao_peaks_w = extract_ao_peaks(s_ao_w, fs_proc, prominence_factor, power=power_exp)

        if len(ao_peaks_w) > 0:
            ao_peaks_global = ao_peaks_w + start_idx_w
            all_ao_peaks.extend(ao_peaks_global)

            if len(ao_peaks_w) > 1:
                ao_intervals_w = np.diff(ao_peaks_w) / fs_proc * 1000
                ao_interval_times_w = (ao_peaks_w[:-1] + ao_peaks_w[1:]) / 2 / fs_proc + window_start + trim_start
                all_ao_intervals.extend(ao_intervals_w)
                all_ao_intervals_times.extend(ao_interval_times_w)

    elapsed_time = time.time() - wall_clock_start
    status_text.text(f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds")
    progress_bar.progress(100)

    if show_sqa_overlay:
        st.info(
            f"SQA skipped windows: {skipped_bad_windows}/{num_windows} "
            f"({(100.0 * skipped_bad_windows / max(1, num_windows)):.1f}%)"
        )

    all_ao_peaks = np.array(all_ao_peaks, dtype=int)
    all_ao_intervals = np.array(all_ao_intervals, dtype=float)
    all_ao_intervals_times = np.array(all_ao_intervals_times, dtype=float)

    if sample_bad_mask_full_record is not None and show_sqa_overlay:
        ao_interval_indices = (all_ao_intervals_times * fs_proc).astype(int)
        ao_good_mask = np.array(
            [
                not sample_bad_mask_full_record[min(idx, len(sample_bad_mask_full_record) - 1)]
                for idx in ao_interval_indices
            ],
            dtype=bool,
        )
        all_ao_intervals = all_ao_intervals[ao_good_mask]
        all_ao_intervals_times = all_ao_intervals_times[ao_good_mask]

    if save_json_output and len(all_ao_peaks) > 0:
        global_peaks = all_ao_peaks + int(trim_start * fs_proc)
        saved_file = save_peaks_to_json(global_peaks, fs_proc, file_label, output_folder)
        st.info(f"AO Peaks saved to: {saved_file}")

    if (
        save_ppg_json_output
        and beat_source == "Detect from PPG raw"
        and not ppg_df.empty
        and len(ppg_vis_peaks_idx) > 0
    ):
        ppg_time_full = ppg_df["time_s"].to_numpy(dtype=float)
        ppg_peak_times = ppg_time_full[ppg_vis_peaks_idx]
        ppg_peak_times = ppg_peak_times[
            (ppg_peak_times >= float(trim_start)) & (ppg_peak_times < float(trim_end))
        ]
        if len(ppg_peak_times) > 0:
            saved_file_ppg = save_peaks_seconds_to_json(
                ppg_peak_times,
                file_label,
                output_folder,
                peak_label="PPG",
            )
            st.info(f"PPG Peaks saved to: {saved_file_ppg}")

    full_time_axis = np.arange(len(scg_proc_full)) / fs_proc + trim_start
    if len(scg_raw) == len(scg_proc_full):
        scg_raw_display = scg_raw
    else:
        scg_raw_display = signal.resample(scg_raw, len(scg_proc_full))
    ppg_intervals_full = np.diff(ppg_peaks_full) / fs_proc * 1000.0 if len(ppg_peaks_full) > 1 else np.array([])
    ppg_interval_times_full = (
        (ppg_peaks_full[:-1] + ppg_peaks_full[1:]) / 2.0 / fs_proc + trim_start
        if len(ppg_peaks_full) > 1
        else np.array([])
    )

    detection_metrics_full = None
    if len(all_ao_peaks) > 0 and len(ppg_peaks_ref_trim) > 0:
        all_ao_peaks_abs = all_ao_peaks + start_idx_global
        detection_metrics_full = compute_detection_metrics(
            all_ao_peaks_abs,
            ppg_peaks_ref_trim,
            fs_proc,
            ref_fs,
            tolerance_seconds=0.2,
        )

    ao_intervals_matched = np.array([])
    ppg_intervals_matched = np.array([])
    ao_centers_s = np.array([])
    ppg_centers_s = np.array([])
    iqr_removed_ao_centers_s = np.array([])
    iqr_removed_ppg_centers_s = np.array([])
    if len(all_ao_peaks) > 1 and len(ppg_peaks_full) > 1:
        (
            ao_intervals_matched,
            ppg_intervals_matched,
            ao_centers_s,
            ppg_centers_s,
            iqr_removed_ao_centers_s,
            iqr_removed_ppg_centers_s,
        ) = match_intervals_by_time(
            all_ao_peaks,
            ppg_peaks_full,
            fs_proc,
            apply_iqr=use_iqr_filter,
        )

    has_ppg_vis = ppg_vis_filtered is not None and not ppg_df.empty
    st.subheader("Signals and Interval Time Series")
    if has_ppg_vis:
        fig_intervals = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Raw SCG with AO Peaks",
                "Processed SCG with AO Peaks and PPG Beats",
                "Processed PPG with Peaks",
                "PPG-PPG and AO-AO Intervals",
            ),
            row_heights=[0.3, 0.3, 0.2, 0.2],
        )
    else:
        fig_intervals = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Raw SCG with AO Peaks",
                "Processed SCG with AO Peaks and PPG Beats",
                "PPG-PPG and AO-AO Intervals",
            ),
            row_heights=[0.35, 0.35, 0.3],
        )

    fig_intervals.add_trace(
        go.Scatter(
            x=full_time_axis,
            y=scg_raw_display,
            mode="lines",
            line=dict(color="gray", width=0.6),
            name="SCG (Raw)",
        ),
        row=1,
        col=1,
    )

    if len(all_ao_peaks) > 0:
        fig_intervals.add_trace(
            go.Scatter(
                x=all_ao_peaks / fs_proc + trim_start,
                y=scg_raw_display[all_ao_peaks],
                mode="markers",
                marker=dict(color="red", size=4, symbol="circle"),
                name="AO Peaks (Raw)",
            ),
            row=1,
            col=1,
        )

    if sample_bad_mask_full_record is not None:
        scg_good = np.where(~sample_bad_mask_full_record, scg_proc_full, np.nan)
        scg_bad = np.where(sample_bad_mask_full_record, scg_proc_full, np.nan)
        fig_intervals.add_trace(
            go.Scatter(
                x=full_time_axis,
                y=scg_good,
                mode="lines",
                line=dict(color="navy", width=0.6),
                name="SCG (Good)",
            ),
            row=2,
            col=1,
        )
        fig_intervals.add_trace(
            go.Scatter(
                x=full_time_axis,
                y=scg_bad,
                mode="lines",
                line=dict(color="red", width=0.9),
                name="SCG (Bad)",
            ),
            row=2,
            col=1,
        )
    else:
        fig_intervals.add_trace(
            go.Scatter(
                x=full_time_axis,
                y=scg_proc_full,
                mode="lines",
                line=dict(color="navy", width=0.6),
                name="SCG",
            ),
            row=2,
            col=1,
        )

    if len(all_ao_peaks) > 0:
        fig_intervals.add_trace(
            go.Scatter(
                x=all_ao_peaks / fs_proc + trim_start,
                y=scg_proc_full[all_ao_peaks],
                mode="markers",
                marker=dict(color="red", size=4, symbol="circle"),
                name="AO Peaks (Processed)",
            ),
            row=2,
            col=1,
        )

    if len(ppg_peaks_full) > 0:
        fig_intervals.add_trace(
            go.Scatter(
                x=ppg_peaks_full / fs_proc + trim_start,
                y=scg_proc_full[ppg_peaks_full],
                mode="markers",
                marker=dict(color="green", size=4, symbol="x"),
                name="PPG Beats",
            ),
            row=2,
            col=1,
        )

    if detection_metrics_full:
        fp_peaks = detection_metrics_full.get("fp_peaks", np.array([], dtype=int)).astype(int)
        fn_times_s = detection_metrics_full.get("fn_times_s", np.array([], dtype=float))
        fp_peaks = fp_peaks - start_idx_global
        fp_peaks = fp_peaks[(fp_peaks >= 0) & (fp_peaks < len(scg_proc_full))]
        if len(fp_peaks) > 0:
            fig_intervals.add_trace(
                go.Scatter(
                    x=full_time_axis[fp_peaks],
                    y=scg_proc_full[fp_peaks],
                    mode="markers",
                    marker=dict(color="#ff8c00", size=6, symbol="triangle-up"),
                    name="False Positives",
                ),
                row=2,
                col=1,
            )
        if len(fn_times_s) > 0:
            for bt in fn_times_s:
                fig_intervals.add_vline(
                    x=float(bt),
                    line_width=1,
                    line_dash="dot",
                    line_color="#6a5acd",
                    row=2,
                    col=1,
                )

    intervals_row = 3
    if has_ppg_vis:
        ppg_time_full = ppg_df["time_s"].to_numpy(dtype=float)
        ppg_mask = (ppg_time_full >= trim_start) & (ppg_time_full <= trim_end)
        ppg_time_trim = ppg_time_full[ppg_mask]
        ppg_filtered_trim = ppg_vis_filtered[ppg_mask]
        if beat_source == "Detect from PPG raw":
            ppg_peaks_in_trim = ppg_vis_peaks_idx[
                (ppg_time_full[ppg_vis_peaks_idx] >= trim_start)
                & (ppg_time_full[ppg_vis_peaks_idx] <= trim_end)
            ]
        else:
            beat_mask = (beat_times_s >= trim_start) & (beat_times_s <= trim_end)
            beat_times_trim = beat_times_s[beat_mask]
            ppg_peaks_in_trim = _map_times_to_indices(ppg_time_full, beat_times_trim)

        fig_intervals.add_trace(
            go.Scatter(
                x=ppg_time_trim,
                y=ppg_filtered_trim,
                mode="lines",
                line=dict(color="#ff4757", width=0.9),
                name="PPG (Filtered)",
            ),
            row=3,
            col=1,
        )
        if len(ppg_peaks_in_trim) > 0:
            fig_intervals.add_trace(
                go.Scatter(
                    x=ppg_time_full[ppg_peaks_in_trim],
                    y=ppg_vis_filtered[ppg_peaks_in_trim],
                    mode="markers",
                    marker=dict(color="#00e5ff", size=5, symbol="circle"),
                    name="PPG Peaks",
                ),
                row=3,
                col=1,
            )
        if use_iqr_filter:
            for center_s in iqr_removed_ao_centers_s:
                fig_intervals.add_vline(
                    x=float(center_s + trim_start),
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(255, 140, 0, 0.6)",
                    row=2,
                    col=1,
                )
            for center_s in iqr_removed_ppg_centers_s:
                fig_intervals.add_vline(
                    x=float(center_s + trim_start),
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(0, 229, 255, 0.6)",
                    row=3,
                    col=1,
                )
        intervals_row = 4

    if len(ppg_interval_times_full) > 0:
        fig_intervals.add_trace(
            go.Scatter(
                x=ppg_interval_times_full,
                y=ppg_intervals_full,
                mode="lines+markers",
                line=dict(color="green", width=1.2),
                marker=dict(size=3),
                name="PPG-PPG Intervals",
            ),
            row=intervals_row,
            col=1,
        )

    if len(all_ao_intervals_times) > 0:
        fig_intervals.add_trace(
            go.Scatter(
                x=all_ao_intervals_times,
                y=all_ao_intervals,
                mode="lines+markers",
                line=dict(color="red", width=1.2),
                marker=dict(size=3),
                name="AO-AO Intervals",
            ),
            row=intervals_row,
            col=1,
        )

    fig_intervals.update_layout(
        height=700,
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(x=1.02, y=0.5, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig_intervals.update_xaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
    fig_intervals.update_yaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
    fig_intervals.update_xaxes(title_text="Time (s)", row=intervals_row, col=1)
    fig_intervals.update_yaxes(title_text="Raw SCG", row=1, col=1)
    fig_intervals.update_yaxes(title_text="Processed SCG", row=2, col=1)
    if has_ppg_vis:
        fig_intervals.update_yaxes(title_text="PPG (Filtered)", row=3, col=1)
    fig_intervals.update_yaxes(title_text="Interval (ms)", row=intervals_row, col=1)
    st.plotly_chart(fig_intervals, width="stretch")

    if len(all_ao_peaks) > 1 and len(ppg_peaks_full) > 1:
        if sample_bad_mask_full_record is not None and show_sqa_overlay:
            ao_center_idx = np.minimum(
                (ao_centers_s * fs_proc).astype(int),
                len(sample_bad_mask_full_record) - 1,
            )
            ppg_center_idx = np.minimum(
                (ppg_centers_s * fs_proc).astype(int),
                len(sample_bad_mask_full_record) - 1,
            )
            good_mask = (
                ~sample_bad_mask_full_record[ao_center_idx]
                & ~sample_bad_mask_full_record[ppg_center_idx]
            )
            ao_intervals_matched = ao_intervals_matched[good_mask]
            ppg_intervals_matched = ppg_intervals_matched[good_mask]

        if len(ao_intervals_matched) == 0:
            st.warning("Not enough matched intervals after filtering.")
        else:
            correlation = (
                np.corrcoef(ao_intervals_matched, ppg_intervals_matched)[0, 1]
                if len(ao_intervals_matched) > 1
                else np.nan
            )
            rmse = np.sqrt(np.mean((ao_intervals_matched - ppg_intervals_matched) ** 2))
            mae = np.mean(np.abs(ao_intervals_matched - ppg_intervals_matched))
            mean_intervals = (ao_intervals_matched + ppg_intervals_matched) / 2
            diff_intervals = ao_intervals_matched - ppg_intervals_matched
            mean_diff = np.mean(diff_intervals)
            std_diff = np.std(diff_intervals)
            paper_metrics = compute_paper_metrics(ao_intervals_matched, ppg_intervals_matched)
            all_ao_peaks_abs = all_ao_peaks + start_idx_global
            ptt_metrics = compute_ptt_metrics(all_ao_peaks_abs, ppg_peaks_ref_trim, fs_proc, ref_fs)

            st.subheader("Full Record Interval Comparison")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Correlation", f"{correlation:.3f}")
            col2.metric("RMSE", f"{rmse:.1f} ms")
            col3.metric("MAE", f"{mae:.1f} ms")
            col4.metric("Bias", f"{mean_diff:.1f} ms")

            if paper_metrics:
                col5, col6, col7, col8, col9 = st.columns(5)
                col5.metric("Mean SCG HR (BPM)", f"{paper_metrics['mean_scg_hr']:.1f}")
                col6.metric("Mean PPG HR (BPM)", f"{paper_metrics['mean_ref_hr']:.1f}")
                col7.metric("ARE", f"{paper_metrics['ARE']:.3f}")
                col8.metric("AAE (BPM)", f"{paper_metrics['AAE']:.2f}")
                col9.metric("AAEP (%)", f"{paper_metrics['AAEP']:.2f}")

                col10, col11, col12 = st.columns(3)
                col10.metric("BA Bias (ms)", f"{paper_metrics['ba_bias']:.1f}")
                col11.metric("BA Upper LOA (ms)", f"{paper_metrics['ba_upper_loa']:.1f}")
                col12.metric("BA Lower LOA (ms)", f"{paper_metrics['ba_lower_loa']:.1f}")

            if use_iqr_filter:
                if iqr_removed_ao_centers_s.size > 0 or iqr_removed_ppg_centers_s.size > 0:
                    with st.expander("IQR-filtered intervals", expanded=False):
                        st.write(
                            f"Removed AO intervals: {len(iqr_removed_ao_centers_s)} | "
                            f"Removed PPG intervals: {len(iqr_removed_ppg_centers_s)}"
                        )
                        if iqr_removed_ao_centers_s.size > 0:
                            st.dataframe(
                                pd.DataFrame({"ao_interval_center_s": iqr_removed_ao_centers_s + trim_start}),
                                width="stretch",
                                hide_index=True,
                            )
                        if iqr_removed_ppg_centers_s.size > 0:
                            st.dataframe(
                                pd.DataFrame({"ppg_interval_center_s": iqr_removed_ppg_centers_s + trim_start}),
                                width="stretch",
                                hide_index=True,
                            )

            if ptt_metrics:
                col13, col14, col15 = st.columns(3)
                col13.metric("Mean PTT (ms)", f"{ptt_metrics['mean_ptt_ms']:.1f}")
                col14.metric("PTT SD (ms)", f"{ptt_metrics['std_ptt_ms']:.1f}")
                col15.metric("PTT vs AO Corr", f"{ptt_metrics['ptt_rr_correlation']:.3f}")

            detection_metrics = detection_metrics_full
            if detection_metrics:
                col10, col11, col12, col13, col14, col15, col16 = st.columns(7)
                col10.metric("TP", f"{detection_metrics['TP']}")
                col11.metric("FP", f"{detection_metrics['FP']}")
                col12.metric("FN", f"{detection_metrics['FN']}")
                col13.metric("SE (%)", f"{detection_metrics['SE']:.1f}")
                col14.metric("P (%)", f"{detection_metrics['P']:.1f}")
                col15.metric("ACC (%)", f"{detection_metrics['ACC']:.1f}")
                col16.metric("DER (%)", f"{detection_metrics['DER']:.1f}")

                with st.expander("Detection errors", expanded=False):
                    fp_times = detection_metrics.get("fp_times_s", np.array([]))
                    fn_times = detection_metrics.get("fn_times_s", np.array([]))
                    fp_peaks = detection_metrics.get("fp_peaks", np.array([]))
                    fn_peaks = detection_metrics.get("fn_peaks", np.array([]))
                    st.write(f"False positives: {len(fp_times)} | False negatives: {len(fn_times)}")
                    if len(fp_times) > 0:
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "fp_sample_idx": fp_peaks,
                                    "fp_time_s": fp_times,
                                }
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    if len(fn_times) > 0:
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "fn_sample_idx": fn_peaks,
                                    "fn_time_s": fn_times,
                                }
                            ),
                            width="stretch",
                            hide_index=True,
                        )

            fig_ba = go.Figure()
            fig_ba.add_trace(
                go.Scatter(
                    x=mean_intervals,
                    y=diff_intervals,
                    mode="markers",
                    marker=dict(size=6, color="blue", opacity=0.6),
                    name="Data points",
                )
            )
            fig_ba.add_hline(
                y=mean_diff,
                line_dash="solid",
                line_color="red",
                line_width=2,
                annotation_text=f"Mean: {mean_diff:.1f} ms",
                annotation_position="right",
            )
            upper_loa = mean_diff + 1.96 * std_diff
            lower_loa = mean_diff - 1.96 * std_diff
            fig_ba.add_hline(
                y=upper_loa,
                line_dash="dash",
                line_color="gray",
                line_width=1.5,
                annotation_text=f"+1.96 SD: {upper_loa:.1f} ms",
                annotation_position="right",
            )
            fig_ba.add_hline(
                y=lower_loa,
                line_dash="dash",
                line_color="gray",
                line_width=1.5,
                annotation_text=f"-1.96 SD: {lower_loa:.1f} ms",
                annotation_position="right",
            )
            fig_ba.update_layout(
                title="Bland-Altman Plot: AO-AO vs PPG-PPG Intervals",
                xaxis_title="Mean of AO-AO and PPG-PPG (ms)",
                yaxis_title="Difference (AO-AO - PPG-PPG) (ms)",
                height=420,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_ba, width="stretch")

            fig_corr = go.Figure()
            fig_corr.add_trace(
                go.Scatter(
                    x=ppg_intervals_matched,
                    y=ao_intervals_matched,
                    mode="markers",
                    marker=dict(size=6, color="green", opacity=0.6),
                    name="Intervals",
                )
            )
            min_val = min(np.min(ppg_intervals_matched), np.min(ao_intervals_matched))
            max_val = max(np.max(ppg_intervals_matched), np.max(ao_intervals_matched))
            fig_corr.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="red", dash="dash", width=2),
                    name="Identity line",
                )
            )
            fig_corr.update_layout(
                title=f"AO-AO vs PPG-PPG Intervals (r = {correlation:.3f})",
                xaxis_title="PPG-PPG Interval (ms)",
                yaxis_title="AO-AO Interval (ms)",
                height=420,
                plot_bgcolor="white",
                hovermode="closest",
            )
            fig_corr.update_xaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
            fig_corr.update_yaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
            st.plotly_chart(fig_corr, width="stretch")

    else:
        st.warning("Not enough peaks detected for interval comparison.")
