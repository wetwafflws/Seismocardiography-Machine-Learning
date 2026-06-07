"""
hvdnet_inference_qt.py — PyQt5 GUI for HVDNet 3-class (AS/MR/Normal) inference.

Supports both Dataset 1 (Cleaned) from Data/ and Dataset 2 (Segmented) from
Subject_Data_Segmented/.  Patients are filtered to the 3-class task:
  - AS only      (AS=1, all other valves 0)
  - MR only      (MR=1, all other valves 0)
  - Normal       (all valve labels 0)
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QComboBox,
    QTabWidget, QScrollArea, QSizePolicy, QGroupBox, QFormLayout,
    QMessageBox, QProgressBar, QTextBrowser, QFrame, QCheckBox,
)
from scipy import signal
import pyqtgraph as pg

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class sCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout1d(p=0.2)
        self.skip_projection = nn.Identity()
        if in_channels != out_channels:
            self.skip_projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip_projection(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        out += identity; out = self.relu(out)
        out = self.pool(out); out = self.dropout(out)
        return out


class SCNN_Module(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, kernel_sizes=(15, 7, 3)):
        super().__init__()
        channels = (base_filters, base_filters // 2, base_filters // 4)
        blocks, c_in = [], in_channels
        for c_out, ks in zip(channels, kernel_sizes):
            blocks.append(sCNN_Block(c_in, c_out, ks))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class LSTM_Module(nn.Module):
    def __init__(self, input_features=16, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return out


class SA_Module(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores = torch.tanh(self.dense(lstm_out))
        weights = F.softmax(scores, dim=1)
        weighted_out = lstm_out * weights
        context_vector = torch.sum(weighted_out, dim=1)
        return context_vector, weights


class HVDNet(nn.Module):
    def __init__(self, num_classes=5, d=32):
        super().__init__()
        self.scnn_x = SCNN_Module(in_channels=1, base_filters=d)
        self.scnn_y = SCNN_Module(in_channels=1, base_filters=d)
        self.scnn_z = SCNN_Module(in_channels=1, base_filters=d)
        self.lstm_x = LSTM_Module(input_features=d // 4, hidden_size=d)
        self.lstm_y = LSTM_Module(input_features=d // 4, hidden_size=d)
        self.lstm_z = LSTM_Module(input_features=d // 4, hidden_size=d)
        self.sa_x = SA_Module(hidden_size=d)
        self.sa_y = SA_Module(hidden_size=d)
        self.sa_z = SA_Module(hidden_size=d)
        self.bn_x = nn.BatchNorm1d(d)
        self.bn_y = nn.BatchNorm1d(d)
        self.bn_z = nn.BatchNorm1d(d)
        self.dropout_sa = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(3 * d, d), nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(p=0.3),
            nn.Linear(d, num_classes),
        )

    def forward(self, x, y, z):
        feat_x = self.scnn_x(x); feat_y = self.scnn_y(y); feat_z = self.scnn_z(z)
        lstm_x = self.lstm_x(feat_x); lstm_y = self.lstm_y(feat_y); lstm_z = self.lstm_z(feat_z)
        ctx_x, attn_x = self.sa_x(lstm_x)
        ctx_y, attn_y = self.sa_y(lstm_y)
        ctx_z, attn_z = self.sa_z(lstm_z)
        ctx_x = self.dropout_sa(self.bn_x(ctx_x))
        ctx_y = self.dropout_sa(self.bn_y(ctx_y))
        ctx_z = self.dropout_sa(self.bn_z(ctx_z))
        concat_vector = torch.cat((ctx_x, ctx_y, ctx_z), dim=1)
        concat_vector = F.dropout(concat_vector, p=0.3, training=self.training)
        logits = self.classifier(concat_vector)
        return logits, (attn_x, attn_y, attn_z)


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_FS = 256
CLASS_NAMES_3 = ["AS", "MR", "N"]
LABEL_COLS = [
    "Moderate or greater MS", "Moderate or greater MR",
    "Moderate or greater AR", "Moderate or greater AS",
    "Moderate or greater TR",
]


def zscore_normalize(values):
    values = np.asarray(values, dtype=float)
    m, s = np.mean(values), np.std(values)
    return np.zeros_like(values) if s < 1e-12 else (values - m) / s


def pad_or_truncate(values, target_len=800):
    v = np.asarray(values, dtype=float)
    if len(v) < target_len:
        return np.pad(v, (0, target_len - len(v)), mode='constant')
    return v[:target_len]


def butter_bandpass(signals_dict, fs, lowcut=1.0, highcut=30.0, order=6):
    b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    return {k: signal.filtfilt(b, a, np.asarray(v, dtype=float)) for k, v in signals_dict.items()}


def build_3beat_segments(r_peaks, signal_length):
    segs = []
    for i in range(len(r_peaks) - 3):
        s, e = int(r_peaks[i]), int(r_peaks[i + 3])
        if 0 <= s < e <= signal_length:
            segs.append({'segment_id': i, 'start_idx': s, 'end_idx': e})
    return segs


def time_str_to_seconds(t):
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def get_original_fs(patient_id):
    if patient_id.startswith('UP-'):
        try:
            if 22 <= int(patient_id.split('-')[1]) <= 30:
                return 512
        except ValueError:
            pass
    return 256


# ── Task mapping for 3-class (AS-only, MR-only, Normal) ─────────────────────

def map_to_3class(label_row):
    ms = int(label_row.get("Moderate or greater MS", 0))
    mr = int(label_row.get("Moderate or greater MR", 0))
    ar = int(label_row.get("Moderate or greater AR", 0))
    as_val = int(label_row.get("Moderate or greater AS", 0))
    tr = int(label_row.get("Moderate or greater TR", 0))

    total = ms + mr + ar + as_val + tr
    if total == 0:
        return 2  # Normal
    if tr == 1 or ms == 1 or ar == 1:
        return None  # excluded
    if (as_val + mr) != 1:
        return None  # multiple or none
    if as_val == 1:
        return 0  # AS
    if mr == 1:
        return 1  # MR
    return None


def load_ground_truth(data_dir="Data"):
    path = os.path.join(data_dir, "ground_truth_labels.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return {row["Patient ID"]: row.to_dict() for _, row in df.iterrows()}


def labels_from_meta(meta):
    """Convert meta.json cardiac_conditions into the 5-label format matching ground_truth_labels.csv columns."""
    label_row = {col: 0 for col in LABEL_COLS}
    conditions = meta.get("cardiac_conditions") or []
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        if condition is None:
            continue
        norm = str(condition).strip()
        if not norm or norm.lower() == "normal":
            continue
        tokens = set(norm.upper().replace("-", " ").replace("/", " ").split())
        if "MS" in tokens or ("MITRAL" in tokens and "STENOSIS" in tokens):
            label_row["Moderate or greater MS"] = 1
        if "MR" in tokens or ("MITRAL" in tokens and "REGURGITATION" in tokens):
            label_row["Moderate or greater MR"] = 1
        if "AR" in tokens or ("AORTIC" in tokens and "REGURGITATION" in tokens):
            label_row["Moderate or greater AR"] = 1
        if "AS" in tokens or ("AORTIC" in tokens and "STENOSIS" in tokens):
            label_row["Moderate or greater AS"] = 1
        if "TR" in tokens or ("TRICUSPID" in tokens and "REGURGITATION" in tokens):
            label_row["Moderate or greater TR"] = 1
    return label_row


def load_segmented_labels(subject_dir):
    """Build labels_lookup for all segmented patients from their _meta.json files."""
    labels = {}
    for f in sorted(glob.glob(os.path.join(subject_dir, "**", "*_meta.json"), recursive=True)):
        pid = os.path.basename(f).replace("_meta.json", "")
        with open(f) as handle:
            meta = json.load(handle)
        labels[pid] = labels_from_meta(meta)
    return labels


# ── Dataset 1 (Cleaned) loader ──────────────────────────────────────────────

def load_cleaned_patient(patient_id, data_dir="Data"):
    csv_path = os.path.join(data_dir, f"Cleaned_{patient_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing Cleaned_{patient_id}.csv")

    df = pd.read_csv(csv_path)
    fs_orig = get_original_fs(patient_id)

    scg_x = df['AccX'].to_numpy(dtype=float)
    scg_y = df['AccY'].to_numpy(dtype=float)
    scg_z = df['AccZ'].to_numpy(dtype=float)
    ecg = df['ECG'].to_numpy(dtype=float)

    if fs_orig == 512:
        n = len(scg_x) // 2
        scg_x = signal.resample(scg_x, n)
        scg_y = signal.resample(scg_y, n)
        scg_z = signal.resample(scg_z, n)
        ecg = signal.resample(ecg, n)

    sig_len = len(scg_x)

    json_path = os.path.join(data_dir, f"{patient_id}-ECG.json")
    peak_times = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        peak_times = data.get("LARA_R_Peaks", list(data.values())[0] if data else [])

    peak_indices = [int(np.round(time_str_to_seconds(ts) * TARGET_FS)) for ts in peak_times]
    peak_indices = [idx for idx in peak_indices if 0 <= idx < sig_len]

    return {
        'signals': {'AccX': scg_x, 'AccY': scg_y, 'AccZ': scg_z, 'ECG': ecg},
        'fs': TARGET_FS, 'signal_length': sig_len,
        'r_peaks_indices': peak_indices, 'dataset': 'Cleaned',
    }


# ── Dataset 2 (Segmented) loader ────────────────────────────────────────────

def load_segmented_patient(patient_id, subject_dir):
    csv_matches = sorted(glob.glob(os.path.join(subject_dir, "**", f"{patient_id}.csv"), recursive=True))
    if not csv_matches:
        raise FileNotFoundError(f"No segmented CSV for {patient_id}")
    csv_path = csv_matches[0]

    meta_matches = sorted(glob.glob(os.path.join(subject_dir, "**", f"{patient_id}_meta.json"), recursive=True))
    meta = {}
    if meta_matches:
        with open(meta_matches[0]) as f:
            meta = json.load(f)

    ppg_matches = sorted(glob.glob(os.path.join(subject_dir, "**", f"{patient_id}-PPG.json"), recursive=True))
    ppg_peak_times = []
    if ppg_matches:
        with open(ppg_matches[0]) as f:
            ppg_peak_times = json.load(f).get("PPG_Peaks", [])

    df = pd.read_csv(csv_path, comment="#")
    mask = df["x_g"].notna()
    scg_x = df.loc[mask, "x_g"].to_numpy(dtype=np.float32)
    scg_y = df.loc[mask, "y_g"].to_numpy(dtype=np.float32)
    scg_z = df.loc[mask, "z_g"].to_numpy(dtype=np.float32)
    ppg_raw = df.loc[mask, "ppg_raw"].to_numpy(dtype=np.float32)

    orig_fs = int(meta.get("sample_rate_scg_hz", TARGET_FS))
    if orig_fs != TARGET_FS:
        n = int(round(len(scg_x) * TARGET_FS / float(orig_fs)))
        scg_x = signal.resample(scg_x, n)
        scg_y = signal.resample(scg_y, n)
        scg_z = signal.resample(scg_z, n)
        ppg_raw = signal.resample(ppg_raw, n)

    sig_len = len(scg_x)
    peak_indices = [int(np.round(time_str_to_seconds(ts) * TARGET_FS)) for ts in ppg_peak_times]
    peak_indices = [idx for idx in peak_indices if 0 <= idx < sig_len]

    return {
        'signals': {'AccX': scg_x, 'AccY': scg_y, 'AccZ': scg_z, 'ECG': ppg_raw},
        'fs': TARGET_FS, 'signal_length': sig_len,
        'r_peaks_indices': peak_indices, 'dataset': 'Segmented',
    }


# ── Device ──────────────────────────────────────────────────────────────────

def get_best_torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_to_device(tensor, device):
    return tensor.to(device, non_blocking=(device.type == "cuda"))


# ═══════════════════════════════════════════════════════════════════════════════
# Styling
# ═══════════════════════════════════════════════════════════════════════════════

BG       = "#f8fafc"
BG_PANEL = "#f1f5f9"
BG_CARD  = "#ffffff"
BG_INPUT = "#ffffff"
BORDER   = "#cbd5e1"
ACCENT   = "#2563eb"
ACCENT2  = "#ef4444"
GREEN    = "#10b981"
AMBER    = "#f59e0b"
MUTED    = "#94a3b8"
TEXT     = "#0f172a"
TEXT_DIM = "#64748b"

COLORS_SCG = [ACCENT, "#7c3aed", GREEN]
FONT_UI = "'Liberation Sans', 'DejaVu Sans', 'Arial', sans-serif"
FONT_MONO = "'Liberation Mono', 'DejaVu Sans Mono', 'Courier New', monospace"

STYLESHEET = f"""
QMainWindow, QWidget {{ background-color: {BG}; color: {TEXT}; font-family: {FONT_UI}; font-size: 11px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG_CARD}; }}
QTabBar::tab {{ background: {BG}; color: {TEXT_DIM}; border: 1px solid {BORDER}; padding: 5px 14px; font-size: 10px; letter-spacing: 1px; }}
QTabBar::tab:selected {{ background: {BG_CARD}; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; margin-top: 8px; padding-top: 4px; font-size: 10px; color: {TEXT_DIM}; letter-spacing: 1px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {TEXT_DIM}; }}
QComboBox {{ background-color: {BG_INPUT}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 3px; padding: 4px 8px; }}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{ background-color: {BG_CARD}; color: {TEXT}; selection-background-color: {BORDER}; }}
QLineEdit {{ background-color: {BG_INPUT}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 3px; padding: 4px 6px; font-family: {FONT_MONO}; }}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QPushButton {{ background-color: {BG_CARD}; color: {ACCENT}; border: 1px solid {ACCENT}; border-radius: 3px; padding: 6px 14px; font-weight: bold; letter-spacing: 1px; font-size: 10px; }}
QPushButton:hover {{ background-color: {ACCENT}; color: {BG}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {MUTED}; }}
QFrame#card {{ background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 4px; }}
QFrame#separator {{ background-color: {BORDER}; max-height: 1px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {BG}; width: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical {{ background: {MUTED}; border-radius: 3px; min-height: 20px; }}
QLabel#section_title {{ color: {ACCENT}; font-size: 11px; font-weight: bold; letter-spacing: 3px; }}
QLabel#pred_value {{ color: {ACCENT}; font-size: 26px; font-weight: bold; font-family: {FONT_MONO}; }}
QLabel#pred_label {{ color: {TEXT_DIM}; font-size: 9px; letter-spacing: 2px; }}
"""


def make_plot(title=""):
    pw = pg.PlotWidget()
    pw.setBackground(BG_PANEL)
    pw.showGrid(x=False, y=True, alpha=0.15)
    for ax in ['left', 'bottom']:
        pw.getAxis(ax).setTextPen(pg.mkPen(TEXT_DIM))
        pw.getAxis(ax).setPen(pg.mkPen(BORDER))
    pw.setClipToView(True)
    pw.setDownsampling(auto=True, mode="peak")
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=True)
    if title:
        pw.setTitle(f'<span style="color:{TEXT_DIM};font-size:9px;letter-spacing:2px;">{title}</span>')
    return pw


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Worker
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceWorker(QThread):
    finished = pyqtSignal(object)
    error_sig = pyqtSignal(str)
    progress = pyqtSignal(int)
    log_sig = pyqtSignal(str)

    def __init__(self, model, patient_data, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.patient_data = patient_data
        self.device = device

    def run(self):
        try:
            fs = self.patient_data['fs']
            self.log_sig.emit("Filtering 1-30 Hz bandpass...")
            filtered = butter_bandpass(self.patient_data['signals'], fs)
            peaks = np.asarray(self.patient_data['r_peaks_indices'], dtype=int)
            self.log_sig.emit(f"Building segments from {len(peaks)} peaks...")
            segments = build_3beat_segments(peaks, self.patient_data['signal_length'])

            if not segments:
                self.error_sig.emit("No valid segments from peak annotations.")
                return

            self.log_sig.emit(f"Running inference on {len(segments)} segments...")
            all_logits, attns_x, attns_y, attns_z = [], [], [], []
            total = len(segments)

            self.model.eval()
            with torch.no_grad():
                for idx, seg in enumerate(segments):
                    s, e = seg['start_idx'], seg['end_idx']
                    sx = pad_or_truncate(zscore_normalize(filtered['AccX'][s:e]), 800)
                    sy = pad_or_truncate(zscore_normalize(filtered['AccY'][s:e]), 800)
                    sz = pad_or_truncate(zscore_normalize(filtered['AccZ'][s:e]), 800)

                    xt = move_to_device(torch.tensor(sx, dtype=torch.float32).unsqueeze(0).unsqueeze(0), self.device)
                    yt = move_to_device(torch.tensor(sy, dtype=torch.float32).unsqueeze(0).unsqueeze(0), self.device)
                    zt = move_to_device(torch.tensor(sz, dtype=torch.float32).unsqueeze(0).unsqueeze(0), self.device)

                    logits, (ax, ay, az) = self.model(xt, yt, zt)
                    all_logits.append(logits.cpu())
                    attns_x.append(ax.cpu())
                    attns_y.append(ay.cpu())
                    attns_z.append(az.cpu())

                    if (idx + 1) % 50 == 0 or idx == total - 1:
                        self.progress.emit(int((idx + 1) / total * 100))

            all_logits = torch.cat(all_logits, dim=0)
            probs = torch.softmax(all_logits, dim=1).numpy()
            mean_probs = probs.mean(axis=0)
            pred_classes = np.argmax(probs, axis=1)
            vote_counts = np.bincount(pred_classes, minlength=probs.shape[1])
            final_pred = int(np.argmax(vote_counts))

            self.log_sig.emit(f"Inference complete: {total} segments processed.")
            self.finished.emit({
                'num_segments': total, 'mean_probabilities': mean_probs,
                'prediction': final_pred, 'probabilities_all': probs,
                'predicted_classes': pred_classes, 'segments': segments,
                'filtered_signals': filtered, 'patient_data': self.patient_data,
                'attentions': (attns_x, attns_y, attns_z),
            })
        except Exception as e:
            self.error_sig.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class HVDNetInferenceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVDNet Inference — 3-class (AS / MR / Normal)")
        self.resize(1400, 900)

        self.model = None
        self.model_meta = None
        self.patient_data = None
        self.inference_result = None
        self.worker = None
        self.device = get_best_torch_device()
        self.labels_lookup = {}
        self._patient_list_cache = []

        self._build_ui()
        self._connect_signals()
        self.log(f"HVDNet Inference ready.  Device: {self.device}")
        self._update_controls()

    # ── UI Build ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar = QWidget()
        self.side = QVBoxLayout(sidebar)
        self.side.setSpacing(12)
        self.side.setContentsMargins(8, 8, 8, 8)
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setMinimumWidth(320)
        sidebar_scroll.setMaximumWidth(420)
        layout.addWidget(sidebar_scroll)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)
        self._build_sidebar()
        self._build_tabs()

    def _build_sidebar(self):
        # ── Model ───────────────────────────────────────────────────────────
        g = QGroupBox("MODEL")
        gl = QVBoxLayout(g)
        self.model_path = QLineEdit(); self.model_path.setPlaceholderText("Path to .pt model...")
        br = QHBoxLayout(); br.addWidget(self.model_path); b = QPushButton("Browse"); br.addWidget(b); self.browse_model_btn = b
        gl.addLayout(br)
        self.load_model_btn = QPushButton("Load Model"); gl.addWidget(self.load_model_btn)
        self.model_info = QLabel("No model loaded")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;padding:6px;background:{BG_PANEL};border-radius:3px;")
        gl.addWidget(self.model_info)
        self.side.addWidget(g)

        # ── Data Source ─────────────────────────────────────────────────────
        g2 = QGroupBox("DATA SOURCE")
        g2l = QVBoxLayout(g2)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Dataset 1 (Cleaned) — Data/", "Dataset 2 (Segmented) — Subject_Data_Segmented/"])
        g2l.addWidget(self.source_combo)

        self.data_dir_edit = QLineEdit("Data")
        browse_data = QPushButton("Browse...")
        dr = QHBoxLayout(); dr.addWidget(self.data_dir_edit); dr.addWidget(browse_data)
        self.browse_data_btn = browse_data
        g2l.addWidget(QLabel("Data directory"))
        g2l.addLayout(dr)

        self.refresh_btn = QPushButton("Refresh Patient List")
        g2l.addWidget(self.refresh_btn)

        self.patient_combo = QComboBox()
        g2l.addWidget(QLabel("Patient"))
        g2l.addWidget(self.patient_combo)
        self.patient_meta = QLabel("No patient selected")
        self.patient_meta.setWordWrap(True)
        self.patient_meta.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;padding:6px;background:{BG_PANEL};border-radius:3px;")
        g2l.addWidget(self.patient_meta)

        self.gt_label = QLabel("")
        self.gt_label.setWordWrap(True)
        self.gt_label.setStyleSheet(f"color:{ACCENT2};font-size:10px;font-weight:bold;")
        g2l.addWidget(self.gt_label)

        self.side.addWidget(g2)

        # ── Inference ───────────────────────────────────────────────────────
        g3 = QGroupBox("INFERENCE")
        g3l = QVBoxLayout(g3)
        self.run_btn = QPushButton("Run Inference"); g3l.addWidget(self.run_btn)
        self.progress = QProgressBar(); g3l.addWidget(self.progress)
        self.status = QLabel("Ready"); self.status.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;"); g3l.addWidget(self.status)
        self.side.addWidget(g3)

        # ── Results Card ────────────────────────────────────────────────────
        card = QFrame(); card.setObjectName("card")
        cl = QVBoxLayout(card); cl.setContentsMargins(14, 10, 14, 10)
        t = QLabel("PREDICTION"); t.setObjectName("pred_label"); cl.addWidget(t, alignment=Qt.AlignCenter)
        self.pred_label = QLabel("--"); self.pred_label.setObjectName("pred_value"); self.pred_label.setAlignment(Qt.AlignCenter); cl.addWidget(self.pred_label)
        self.conf_label = QLabel(""); self.conf_label.setAlignment(Qt.AlignCenter); self.conf_label.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;font-weight:bold;"); cl.addWidget(self.conf_label)
        self.probs_label = QLabel(""); self.probs_label.setWordWrap(True); self.probs_label.setStyleSheet(f"color:{TEXT};font-size:10px;padding:6px;background:{BG_PANEL};border-radius:3px;"); cl.addWidget(self.probs_label)
        self.side.addWidget(card)
        self.side.addStretch()

    def _build_tabs(self):
        self.res_tab = QWidget(); self.sig_tab = QWidget(); self.log_tab = QWidget()
        self.tabs.addTab(self.res_tab, "Results")
        self.tabs.addTab(self.sig_tab, "Signals")
        self.tabs.addTab(self.log_tab, "Logs")

        # Results
        rl = QVBoxLayout(self.res_tab)
        rl.setContentsMargins(10, 10, 10, 10)
        self.results_browser = QTextBrowser()
        self.results_browser.setReadOnly(True)
        self.results_browser.setStyleSheet(f"background:{BG_CARD};border:1px solid {BORDER};border-radius:4px;padding:8px;")
        rl.addWidget(self.results_browser)

        # Signals
        sl = QVBoxLayout(self.sig_tab)
        sl.setContentsMargins(10, 10, 10, 10)
        self.gw = pg.GraphicsLayoutWidget()
        self.gw.setBackground(BG_PANEL)
        sl.addWidget(self.gw)

        self.ax = [self.gw.addPlot(row=i, col=0) for i in range(4)]
        labels = ['AccX', 'AccY', 'AccZ', 'PPG']
        for i, p in enumerate(self.ax):
            p.setLabel('left', labels[i])
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setClipToView(True)
        self.ax[3].setLabel('bottom', 'Samples')
        for i in range(1, 4):
            self.ax[i].setXLink(self.ax[0])

        # Logs
        ll = QVBoxLayout(self.log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color:#1e1e1e;color:#00ff00;font-family:monospace;")
        ll.addWidget(self.log_text)

    def _connect_signals(self):
        self.browse_model_btn.clicked.connect(self._browse_model)
        self.load_model_btn.clicked.connect(self._load_model)
        self.browse_data_btn.clicked.connect(self._browse_data_dir)
        self.refresh_btn.clicked.connect(self._refresh_patients)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.patient_combo.currentIndexChanged.connect(self._on_patient_changed)
        self.run_btn.clicked.connect(self._run_inference)

    # ── Actions ─────────────────────────────────────────────────────────────

    def log(self, msg):
        self.log_text.append(msg)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Model (*.pt *.pth)")
        if p:
            self.model_path.setText(p)

    def _load_model(self):
        path = self.model_path.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Model file not found.")
            return
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(payload, dict) and 'model_state_dict' in payload:
                sd = payload['model_state_dict']
                nc = payload.get('num_classes', 3)
                dval = payload.get('d', 32)
                task = payload.get('task_name', '3-class (AS/MR/N)')
                cnames = payload.get('class_names', CLASS_NAMES_3)
            else:
                sd = payload; nc = 3; dval = 32; task = '3-class (AS/MR/N)'; cnames = CLASS_NAMES_3

            model = HVDNet(num_classes=nc, d=dval)
            model.load_state_dict(sd)
            model.to(self.device)
            model.eval()
            self.model = model
            self.model_meta = {'num_classes': nc, 'd': dval, 'task_name': task, 'class_names': cnames}
            self.model_info.setText(f"Task: {task}\nClasses: {cnames}\nd={dval}")
            self.log(f"Model loaded: {path}  |  {task}  |  {cnames}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load model:\n{e}")
            self.log(f"[ERROR] {e}")
        self._update_controls()

    def _browse_data_dir(self):
        p = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if p:
            self.data_dir_edit.setText(p)
            self._refresh_patients()

    def _on_source_changed(self):
        idx = self.source_combo.currentIndex()
        self.data_dir_edit.setText("Data" if idx == 0 else "Subject_Data_Segmented")
        self._refresh_patients()

    def _refresh_patients(self):
        is_cleaned = self.source_combo.currentIndex() == 0
        data_dir = self.data_dir_edit.text().strip()

        if not os.path.exists(data_dir):
            self.patient_combo.clear()
            self._patient_list_cache = []
            self.status.setText("Directory not found")
            return

        # Load ground truth for filtering
        if is_cleaned:
            self.labels_lookup = load_ground_truth(data_dir)
        else:
            self.labels_lookup = load_segmented_labels(data_dir)

        # Find patients
        patient_ids = []
        if is_cleaned:
            for f in sorted(glob.glob(os.path.join(data_dir, "Cleaned_*.csv"))):
                pid = os.path.basename(f).replace("Cleaned_", "").replace(".csv", "")
                patient_ids.append(pid)
        else:
            seen = set()
            for f in sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)):
                bn = os.path.splitext(os.path.basename(f))[0]
                if not bn.endswith("_meta") and bn not in seen:
                    seen.add(bn)
                    patient_ids.append(bn)

        # Filter to 3-class eligible
        eligible = []
        for pid in patient_ids:
            row = self.labels_lookup.get(pid)
            if row is not None:
                cls = map_to_3class(row)
                if cls is not None:
                    eligible.append((pid, cls))
            else:
                eligible.append((pid, None))

        self._patient_list_cache = eligible
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        label_map = {0: "AS", 1: "MR", 2: "N"}
        for pid, cls in eligible:
            tag = f" [{label_map[cls]}]" if cls is not None else " [?]"
            self.patient_combo.addItem(pid + tag, pid)
        self.patient_combo.blockSignals(False)

        self.log(f"Found {len(eligible)} patients in {data_dir} ({sum(1 for _, c in eligible if c is not None)} labeled)")
        if self.patient_combo.count():
            self.patient_combo.setCurrentIndex(0)
            self._on_patient_changed()
        self._update_controls()

    def _on_patient_changed(self):
        pid = self.patient_combo.currentData()
        if not pid:
            self.patient_meta.setText("No patient selected")
            self.gt_label.setText("")
            return

        row = self.labels_lookup.get(pid)
        if row:
            cls = map_to_3class(row)
            label_map = {0: "AS-only", 1: "MR-only", 2: "Normal"}
            if cls is not None:
                self.gt_label.setText(f"Ground Truth: {label_map[cls]}")
                self.gt_label.setStyleSheet(f"color:{GREEN};font-size:10px;font-weight:bold;")
            else:
                # Show what valves are present
                parts = []
                for col, name in zip(LABEL_COLS, ["MS", "MR", "AR", "AS", "TR"]):
                    if int(row.get(col, 0)) == 1:
                        parts.append(name)
                self.gt_label.setText(f"Ground Truth: {', '.join(parts)} (excluded)")
                self.gt_label.setStyleSheet(f"color:{ACCENT2};font-size:10px;font-weight:bold;")
        else:
            self.gt_label.setText("No ground truth label found")
            self.gt_label.setStyleSheet(f"color:{AMBER};font-size:10px;")

        # Show metadata from source
        is_cleaned = self.source_combo.currentIndex() == 0
        data_dir = self.data_dir_edit.text().strip()
        text = f"Source: {'Cleaned' if is_cleaned else 'Segmented'}\n"
        if row:
            text += f"Labels: MS={row.get('Moderate or greater MS',0)} MR={row.get('Moderate or greater MR',0)} AR={row.get('Moderate or greater AR',0)} AS={row.get('Moderate or greater AS',0)} TR={row.get('Moderate or greater TR',0)}"
        else:
            meta_file = sorted(glob.glob(os.path.join(data_dir, "**", f"{pid}_meta.json"), recursive=True))
            if meta_file:
                with open(meta_file[0]) as f:
                    m = json.load(f)
                text += f"Initials: {m.get('patient_initials','?')}  Age: {m.get('age','?')}  Sex: {m.get('sex','?')}"
                conds = m.get('cardiac_conditions', [])
                text += f"\nConditions: {', '.join(conds) if isinstance(conds,list) else conds}"
            else:
                text += "No metadata file"
        self.patient_meta.setText(text)

    def _run_inference(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "Load a model first.")
            return
        pid = self.patient_combo.currentData()
        if not pid:
            QMessageBox.warning(self, "Error", "Select a patient first.")
            return

        is_cleaned = self.source_combo.currentIndex() == 0
        data_dir = self.data_dir_edit.text().strip()
        try:
            if is_cleaned:
                patient_data = load_cleaned_patient(pid, data_dir)
            else:
                patient_data = load_segmented_patient(pid, data_dir)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load patient data:\n{e}")
            return

        self.patient_data = patient_data
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Running inference...")
        self.pred_label.setText("--"); self.conf_label.setText(""); self.probs_label.setText("")

        self.worker = InferenceWorker(self.model, patient_data, self.device)
        self.worker.log_sig.connect(self.log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.error_sig.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, result):
        self.inference_result = result
        cnames = self.model_meta['class_names']
        probs = result['mean_probabilities']
        pred = result['prediction']
        ns = result['num_segments']

        conf = probs[pred] * 100
        self.pred_label.setText(cnames[pred])
        self.conf_label.setText(f"{conf:.1f}% confidence  ({ns} segments)")
        parts = [f"{n}: {p * 100:.1f}%" for n, p in zip(cnames, probs)]
        self.probs_label.setText(" | ".join(parts))
        self.status.setText(f"Done — {ns} segments")

        # Refresh GT display
        self._on_patient_changed()
        self._update_results_browser(result)
        self._update_plots(result)
        self._update_controls()

    def _on_error(self, msg):
        self.status.setText("Error"); self.log(f"[ERROR] {msg}"); self._update_controls()

    def _update_results_browser(self, result):
        cnames = self.model_meta['class_names']
        probs = result['mean_probabilities']
        pred = result['prediction']
        ns = result['num_segments']
        all_probs = result['probabilities_all']
        data = result['patient_data']

        # Compare to ground truth
        gt_str = ""
        pid = data.get('patient_id', '')
        row = self.labels_lookup.get(pid)
        if row:
            cls = map_to_3class(row)
            label_map = {0: "AS", 1: "MR", 2: "Normal"}
            if cls is not None:
                match = "✓" if cls == pred else "✗"
                gt_str = f"<tr><td style='padding:6px;font-weight:bold;'>Ground Truth:</td><td>{label_map[cls]} {match}</td></tr>"

        probs_rows = ""
        for n, p in zip(cnames, probs):
            pct = p * 100
            color = GREEN if n == "N" or n == "Normal" else ACCENT
            probs_rows += f"""
            <tr><td style="padding:4px;font-weight:bold;">{n}</td>
            <td style="padding:4px;width:70%;"><div style="background:{BG_PANEL};border-radius:3px;height:18px;width:100%;"><div style="background:{color};border-radius:3px;height:18px;width:{pct}%;"></div></div></td>
            <td style="padding:4px;font-family:monospace;text-align:right;">{pct:.1f}%</td></tr>"""

        # Vote distribution
        votes = np.bincount(result['predicted_classes'], minlength=len(cnames))
        vote_rows = ""
        for n, v in zip(cnames, votes):
            vote_rows += f"<tr><td style='padding:4px;'>{n}</td><td style='padding:4px;font-family:monospace;text-align:right;'>{v}/{ns}</td></tr>"

        html = f"""
        <div style="font-family:sans-serif;padding:10px;">
            <h2 style="color:{ACCENT};">Inference Results</h2>
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:6px;font-weight:bold;">Patient:</td><td>{data.get('patient_id','')}</td></tr>
                <tr><td style="padding:6px;font-weight:bold;">Dataset:</td><td>{data.get('dataset','')}</td></tr>
                <tr><td style="padding:6px;font-weight:bold;">Segments:</td><td>{ns}</td></tr>
                <tr><td style="padding:6px;font-weight:bold;">Device:</td><td>{self.device}</td></tr>
                {gt_str}
            </table>
            <h3 style="color:{TEXT};">Per-class probabilities</h3>
            <table style="width:100%;border-collapse:collapse;">{probs_rows}</table>
            <h3 style="color:{TEXT};">Segment vote distribution</h3>
            <table style="width:100%;border-collapse:collapse;">{vote_rows}</table>
        </div>"""
        self.results_browser.setHtml(html)

    def _update_plots(self, result):
        filtered = result['filtered_signals']
        data = result['patient_data']
        for p in self.ax: p.clear()
        acc = [filtered['AccX'], filtered['AccY'], filtered['AccZ'], data['signals']['ECG']]
        t = np.arange(len(acc[0]))
        for i, (vals, color) in enumerate(zip(acc, [COLORS_SCG[0], COLORS_SCG[1], COLORS_SCG[2], ACCENT2])):
            self.ax[i].plot(t, vals, pen=pg.mkPen(color, width=1))
        peaks = np.asarray(data['r_peaks_indices'], dtype=int)
        for peak in peaks[:200]:
            self.ax[3].addLine(x=float(peak), pen=pg.mkPen((255, 255, 0, 80), width=1))
        self.ax[0].setTitle(f'<span style="color:{TEXT_DIM};font-size:10px;">Patient {data.get("patient_id","")} — Filtered Signals</span>')
        self.ax[0].setXRange(0, min(2560, len(acc[0])), padding=0.01)
        self.tabs.setCurrentWidget(self.sig_tab)

    def _update_controls(self):
        running = self.worker is not None and self.worker.isRunning()
        self.run_btn.setEnabled(self.model is not None and self.patient_combo.currentData() is not None and not running)
        self.run_btn.setText("Running..." if running else "Run Inference")
        self.load_model_btn.setEnabled(bool(self.model_path.text().strip()))
        self.refresh_btn.setEnabled(bool(self.data_dir_edit.text().strip()))


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    w = HVDNetInferenceWindow()
    w.show()
    sys.exit(app.exec_())
