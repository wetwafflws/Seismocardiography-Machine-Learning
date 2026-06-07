import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.stats import zscore


DEFAULT_UNHEALTHY_BASE_DIR = "Data"
LABELS_FILE_NAME = "ground_truth_labels.csv"

CLASS_COLUMNS = [
    "Moderate or greater MS",
    "Moderate or greater MR",
    "Moderate or greater AR",
    "Moderate or greater AS",
    "Moderate or greater TR",
]
CLASS_SHORT = ["MS", "MR", "AR", "AS", "TR"]


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


def load_unhealthy_labels(base_dir):
    labels_file = os.path.join(base_dir, LABELS_FILE_NAME)
    if not os.path.exists(labels_file):
        return pd.DataFrame()
    labels_df = pd.read_csv(labels_file)
    labels_df.columns = labels_df.columns.str.strip().str.replace("\ufeff", "")
    labels_df.set_index("Patient ID", inplace=True)
    return labels_df


def get_class_label(row):
    active = [short for short, col in zip(CLASS_SHORT, CLASS_COLUMNS) if row.get(col, 0) == 1]
    if not active:
        return "Normal"
    return "+".join(active)


def build_class_map(labels_df):
    class_map = {}
    for patient_id, row in labels_df.iterrows():
        class_label = get_class_label(row)
        class_map.setdefault(class_label, []).append(patient_id)
    for key in class_map:
        class_map[key] = sorted(class_map[key])
    return dict(sorted(class_map.items()))


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

    df_full["scg_z_filt"] = butter_bandpass_filter(df_full["Accel_Z"], 1.0, 40.0, target_fs)
    df_full["scg_z_norm"] = zscore(df_full["scg_z_filt"].to_numpy())
    df_full["ecg_filt"] = butter_bandpass_filter(df_full["ECG"], 1.0, 40.0, target_fs)

    signals = np.zeros((len(df_full), 4))
    signals[:, 0] = df_full["ecg_filt"].to_numpy()
    signals[:, 3] = df_full["scg_z_filt"].to_numpy()

    return signals, r_peak_indices, r_peaks_seconds, target_fs


def get_signal_channel(signals, channel_name):
    if channel_name == "ECG":
        return signals[:, 0]
    if channel_name == "SCG_Z":
        return signals[:, 3]
    raise ValueError(f"Unsupported channel: {channel_name}")


def extract_segment(signal_in, fs, start_s, duration_s):
    start_idx = int(start_s * fs)
    end_idx = int((start_s + duration_s) * fs)
    start_idx = max(0, start_idx)
    end_idx = min(len(signal_in), end_idx)
    return signal_in[start_idx:end_idx], start_idx, end_idx


def extract_segment_by_indices(signal_in, start_idx, end_idx):
    start_idx = max(0, int(start_idx))
    end_idx = min(len(signal_in), int(end_idx))
    if end_idx <= start_idx:
        return np.array([]), start_idx, end_idx
    return signal_in[start_idx:end_idx], start_idx, end_idx


def build_r_peak_segment(r_peaks, fs, beat_index, beats_count, pre_ms, post_ms):
    if len(r_peaks) < 2:
        return 0, 0, False
    beat_index = int(np.clip(beat_index, 0, len(r_peaks) - 2))
    end_beat = int(np.clip(beat_index + beats_count, 1, len(r_peaks) - 1))
    pre_samples = int((pre_ms / 1000.0) * fs)
    post_samples = int((post_ms / 1000.0) * fs)
    start_idx = r_peaks[beat_index] + pre_samples
    end_idx = r_peaks[end_beat] + post_samples
    return start_idx, end_idx, end_idx > start_idx


def rms_normalize(signal_in):
    rms = np.sqrt(np.mean(np.square(signal_in)))
    if rms <= 1e-10:
        return signal_in
    return signal_in / rms


def _detrend(signal_in):
    """Remove linear trend from signal."""
    if len(signal_in) < 2:
        return signal_in
    x_axis = np.arange(len(signal_in))
    coeffs = np.polyfit(x_axis, signal_in, 1)
    trend = np.polyval(coeffs, x_axis)
    return signal_in - trend


def _welch_per_segment(x, y, fs, nperseg, noverlap, window="hann"):
    """Compute cross-spectrum and auto-power spectra using Welch's method manually."""
    win = signal.get_window(window, nperseg)
    step = nperseg - noverlap

    n_segments = (len(x) - noverlap) // step

    # Pre-compute segment indices
    starts = np.arange(0, len(x) - nperseg + 1, step)
    starts = starts[:n_segments]

    if len(starts) == 0:
        return np.array([]), np.array([]), np.array([])

    n_freq = nperseg // 2 + 1
    pxx_sum = np.zeros(n_freq, dtype=np.float64)
    pyy_sum = np.zeros(n_freq, dtype=np.float64)
    pxy_sum = np.zeros(n_freq, dtype=np.complex128)

    for start in starts:
        xs = x[start : start + nperseg] * win
        ys = y[start : start + nperseg] * win

        xs = _detrend(xs)
        ys = _detrend(ys)

        X = np.fft.rfft(xs)
        Y = np.fft.rfft(ys)

        pxx_sum += (X.real ** 2 + X.imag ** 2).real
        pyy_sum += (Y.real ** 2 + Y.imag ** 2).real
        pxy_sum += X * np.conj(Y)

    n_avged = len(starts)
    pxx_avg = pxx_sum / n_avged
    pyy_avg = pyy_sum / n_avged
    pxy_avg = pxy_sum / n_avged

    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, pxx_avg, pyy_avg, pxy_avg


def compute_coherence(x, y, fs, low_hz=1.0, high_hz=40.0):
    min_len = min(len(x), len(y))
    if min_len < 4:
        return np.array([]), np.array([])

    x = x[:min_len]
    y = y[:min_len]

    x = rms_normalize(x)
    y = rms_normalize(y)

    nperseg = min(int(fs * 2), max(8, min_len // 4))
    if min_len // nperseg < 4:
        return np.array([]), np.array([])

    noverlap = nperseg // 2

    # Detrend full signals before segmenting
    x = _detrend(x)
    y = _detrend(y)

    result = _welch_per_segment(x, y, fs, nperseg, noverlap, window="hann")
    if len(result[0]) == 0:
        return np.array([]), np.array([])

    f, pxx, pyy, pxy = result

    # Coherence = |Pxy|^2 / (Pxx * Pyy)
    # Avoid division by zero
    denom = pxx * pyy
    eps = 1e-30
    cxy = np.where(denom > eps, np.abs(pxy) ** 2 / denom, 0.0)
    cxy = np.clip(cxy, 0.0, 1.0)

    mask = (f >= low_hz) & (f <= high_hz)
    return f[mask], cxy[mask]


st.set_page_config(page_title="Cross Spectral Coherence", layout="wide")

st.title("Cross Spectral Coherence Viewer")
st.markdown("Select two subjects from the same class and compare their segments.")

base_dir = st.text_input("Dataset Base Folder", value=DEFAULT_UNHEALTHY_BASE_DIR)

if not os.path.exists(base_dir):
    st.error(f"Dataset base folder not found: {base_dir}")
    st.stop()

labels_df = load_unhealthy_labels(base_dir)
if labels_df.empty:
    st.error("Labels file not found or empty. Ensure Data/ground_truth_labels.csv exists.")
    st.stop()

class_map = build_class_map(labels_df)
class_label = st.selectbox("Class", options=list(class_map.keys()))
patients = class_map.get(class_label, [])

if len(patients) < 2:
    st.warning("Not enough subjects in this class. Pick another class.")
    st.stop()

left_col, right_col = st.columns(2)

with left_col:
    subject_a = st.selectbox("Subject A", options=patients, index=0)
with right_col:
    subject_b = st.selectbox("Subject B", options=patients, index=1)

if subject_a == subject_b:
    st.warning("Subject A and Subject B are the same. Pick two different subjects to avoid identical signals.")

selection_mode = st.radio("Segment Selection", ["Time", "ECG R-peak aligned"], horizontal=True)
channel = st.selectbox("Coherence Channel", options=["SCG_Z", "ECG"], index=0)

seg_col_a, seg_col_b = st.columns(2)

try:
    signals_a, r_peaks_a, _, fs_a = load_unhealthy_patient_data(subject_a, base_dir)
    signals_b, r_peaks_b, _, fs_b = load_unhealthy_patient_data(subject_b, base_dir)
except Exception as exc:
    st.error(f"Failed to load selected subjects: {exc}")
    st.stop()

if fs_a != fs_b:
    st.warning(f"Sampling rates differ: {fs_a} Hz vs {fs_b} Hz. Coherence uses {fs_a} Hz.")

fs = fs_a

ecg_a = get_signal_channel(signals_a, "ECG")
ecg_b = get_signal_channel(signals_b, "ECG")
scg_a = get_signal_channel(signals_a, "SCG_Z")
scg_b = get_signal_channel(signals_b, "SCG_Z")

coh_signal_a = scg_a if channel == "SCG_Z" else ecg_a
coh_signal_b = scg_b if channel == "SCG_Z" else ecg_b

max_duration_a = len(coh_signal_a) / fs
max_duration_b = len(coh_signal_b) / fs

with seg_col_a:
    st.subheader("Segment A")
    if selection_mode == "Time":
        duration_a = st.slider("Duration A (s)", min_value=2.0, max_value=min(20.0, max_duration_a), value=10.0, step=0.5)
        start_a = st.slider("Start A (s)", min_value=0.0, max_value=max(0.0, max_duration_a - duration_a), value=0.0, step=0.5)
        r_beat_index_a = 0
        beats_count_a = 1
        pre_ms_a = -100.0
        post_ms_a = 300.0
    else:
        max_start_a = max(0, len(r_peaks_a) - 2)
        r_beat_index_a = st.slider("R peak index A", min_value=0, max_value=max_start_a, value=0, step=1)
        max_beats_a = max(1, len(r_peaks_a) - r_beat_index_a - 1)
        beats_count_a = st.slider("Beats count A", min_value=1, max_value=min(10, max_beats_a), value=min(3, max_beats_a), step=1)
        pre_ms_a = st.slider("Pre R offset A (ms)", min_value=-300.0, max_value=0.0, value=-100.0, step=10.0)
        post_ms_a = st.slider("Post R offset A (ms)", min_value=0.0, max_value=600.0, value=300.0, step=10.0)
        duration_a = 0.0
        start_a = 0.0

with seg_col_b:
    st.subheader("Segment B")
    if selection_mode == "Time":
        duration_b = st.slider("Duration B (s)", min_value=2.0, max_value=min(20.0, max_duration_b), value=10.0, step=0.5)
        start_b = st.slider("Start B (s)", min_value=0.0, max_value=max(0.0, max_duration_b - duration_b), value=0.0, step=0.5)
        r_beat_index_b = 0
        beats_count_b = 1
        pre_ms_b = -100.0
        post_ms_b = 300.0
    else:
        max_start_b = max(0, len(r_peaks_b) - 2)
        r_beat_index_b = st.slider("R peak index B", min_value=0, max_value=max_start_b, value=0, step=1)
        max_beats_b = max(1, len(r_peaks_b) - r_beat_index_b - 1)
        beats_count_b = st.slider("Beats count B", min_value=1, max_value=min(10, max_beats_b), value=min(3, max_beats_b), step=1)
        pre_ms_b = st.slider("Pre R offset B (ms)", min_value=-300.0, max_value=0.0, value=-100.0, step=10.0)
        post_ms_b = st.slider("Post R offset B (ms)", min_value=0.0, max_value=600.0, value=300.0, step=10.0)
        duration_b = 0.0
        start_b = 0.0

low_hz = st.slider("Coherence Low Hz", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
high_hz = st.slider("Coherence High Hz", min_value=10.0, max_value=80.0, value=40.0, step=5.0)

if selection_mode == "Time":
    segment_a, a_start_idx, a_end_idx = extract_segment(coh_signal_a, fs, start_a, duration_a)
    segment_b, b_start_idx, b_end_idx = extract_segment(coh_signal_b, fs, start_b, duration_b)
else:
    if len(r_peaks_a) < 2 or len(r_peaks_b) < 2:
        st.warning("R peak annotations are missing or too short for one of the subjects.")
        st.stop()
    start_idx_a, end_idx_a, ok_a = build_r_peak_segment(
        r_peaks_a, fs, r_beat_index_a, beats_count_a, pre_ms_a, post_ms_a
    )
    start_idx_b, end_idx_b, ok_b = build_r_peak_segment(
        r_peaks_b, fs, r_beat_index_b, beats_count_b, pre_ms_b, post_ms_b
    )
    if not ok_a or not ok_b:
        st.warning("R-peak aligned segment bounds are invalid. Adjust beat index or offsets.")
        st.stop()
    segment_a, a_start_idx, a_end_idx = extract_segment_by_indices(coh_signal_a, start_idx_a, end_idx_a)
    segment_b, b_start_idx, b_end_idx = extract_segment_by_indices(coh_signal_b, start_idx_b, end_idx_b)

if len(segment_a) < 2 or len(segment_b) < 2:
    st.warning("Segments are too short. Increase duration or adjust start time.")
    st.stop()

min_len = min(len(segment_a), len(segment_b))
segment_a = segment_a[:min_len]
segment_b = segment_b[:min_len]

f, cxy = compute_coherence(segment_a, segment_b, fs, low_hz=low_hz, high_hz=high_hz)

st.subheader("Coherence")
if len(f) == 0:
    st.warning("Segments are too short or identical for a stable coherence estimate.")
    st.stop()
fig = go.Figure()
fig.add_trace(go.Scatter(x=f, y=cxy, mode="lines", line=dict(color="#1f77b4", width=2), name="Coherence"))
fig.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="Coherence",
    yaxis=dict(range=[0, 1]),
    height=400,
    plot_bgcolor="white",
    margin=dict(t=40, b=40),
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")

st.plotly_chart(fig, width='stretch', key="coherence_plot")

st.subheader("Segment Preview")

preview_col_a, preview_col_b = st.columns(2)

ecg_seg_a, _, _ = extract_segment_by_indices(ecg_a, a_start_idx, a_end_idx)
scg_seg_a, _, _ = extract_segment_by_indices(scg_a, a_start_idx, a_end_idx)
ecg_seg_b, _, _ = extract_segment_by_indices(ecg_b, b_start_idx, b_end_idx)
scg_seg_b, _, _ = extract_segment_by_indices(scg_b, b_start_idx, b_end_idx)

with preview_col_a:
    st.markdown(f"**{subject_a}**")
    t_a = np.arange(len(ecg_seg_a)) / fs
    fig_a = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["ECG", "SCG Z"])
    fig_a.add_trace(go.Scatter(x=t_a, y=ecg_seg_a, mode="lines", name="ECG", line=dict(width=1.2)), row=1, col=1)
    fig_a.add_trace(go.Scatter(x=t_a, y=scg_seg_a, mode="lines", name="SCG Z", line=dict(width=1.2)), row=2, col=1)
    fig_a.update_layout(height=350, plot_bgcolor="white", margin=dict(t=40, b=40))
    fig_a.update_xaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
    fig_a.update_yaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
    st.plotly_chart(fig_a, width='stretch', key="preview_a")

with preview_col_b:
    st.markdown(f"**{subject_b}**")
    t_b = np.arange(len(ecg_seg_b)) / fs
    fig_b = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["ECG", "SCG Z"])
    fig_b.add_trace(go.Scatter(x=t_b, y=ecg_seg_b, mode="lines", name="ECG", line=dict(width=1.2)), row=1, col=1)
    fig_b.add_trace(go.Scatter(x=t_b, y=scg_seg_b, mode="lines", name="SCG Z", line=dict(width=1.2)), row=2, col=1)
    fig_b.update_layout(height=350, plot_bgcolor="white", margin=dict(t=40, b=40))
    fig_b.update_xaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
    fig_b.update_yaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
    st.plotly_chart(fig_b, width='stretch', key="preview_b")

st.subheader("Coherence Segments")
time_axis = np.arange(min_len) / fs

fig_segments = go.Figure()
fig_segments.add_trace(
    go.Scatter(x=time_axis, y=segment_a, mode="lines", name=f"{subject_a} ({channel})", line=dict(width=1.5))
)
fig_segments.add_trace(
    go.Scatter(x=time_axis, y=segment_b, mode="lines", name=f"{subject_b} ({channel})", line=dict(width=1.5))
)
fig_segments.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    height=350,
    plot_bgcolor="white",
    margin=dict(t=40, b=40),
)
fig_segments.update_xaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")
fig_segments.update_yaxes(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)")

st.plotly_chart(fig_segments, width='stretch', key="segments_plot")

st.caption(
    f"Class: {class_label} | Subjects: {subject_a}, {subject_b} | "
    f"Segment lengths: {len(segment_a) / fs:.2f}s, {len(segment_b) / fs:.2f}s | "
    f"Range: {low_hz:.1f}-{high_hz:.1f} Hz"
)
