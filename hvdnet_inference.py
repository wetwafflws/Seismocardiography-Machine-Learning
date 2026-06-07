"""
hvdnet_inference.py — Run trained HVDNet inference on segmented SCG data.

Takes a saved model (.pt) from machinelearning_testing.py and runs inference
on subject data produced by raw_svmd_qt.py (Subject_Data_Segmented/ format).

Usage:
    # Single patient
    python hvdnet_inference.py --model model.pt --patient_id CP-01

    # All patients in Subject_Data_Segmented
    python hvdnet_inference.py --model model.pt --subject_dir Subject_Data_Segmented

    # Specific subject directory
    python hvdnet_inference.py --model model.pt --subject_dir Subject_Data_Segmented/2026-05-28
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Model Architecture (mirrors machinelearning_testing.py) ──────────────────

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
        blocks = []
        c_in = in_channels
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
        feat_x = self.scnn_x(x)
        feat_y = self.scnn_y(y)
        feat_z = self.scnn_z(z)
        lstm_x = self.lstm_x(feat_x)
        lstm_y = self.lstm_y(feat_y)
        lstm_z = self.lstm_z(feat_z)
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


# ── Preprocessing helpers (mirrors machinelearning_testing.py) ───────────────

def zscore_normalize(values):
    values = np.asarray(values, dtype=float)
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val < 1e-12:
        return np.zeros_like(values)
    return (values - mean_val) / std_val


def pad_or_truncate(values, target_len=800):
    values = np.asarray(values, dtype=float)
    if len(values) < target_len:
        return np.pad(values, (0, target_len - len(values)), mode='constant')
    if len(values) > target_len:
        return values[:target_len]
    return values


def apply_zero_phase_butterworth(signals_dict, fs, lowcut=1.0, highcut=30.0, order=6):
    nyquist = fs / 2.0
    b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    filtered = {}
    for name, values in signals_dict.items():
        filtered[name] = signal.filtfilt(b, a, np.asarray(values, dtype=float))
    return filtered


def build_rpeak_segments(r_peaks, signal_length):
    segments = []
    for i in range(len(r_peaks) - 3):
        start_idx = int(r_peaks[i])
        end_idx = int(r_peaks[i + 3])
        if 0 <= start_idx < end_idx <= signal_length:
            segments.append({
                'segment_id': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_peak_number': i,
                'end_peak_number': i + 3,
            })
    return segments


# ── Data loading helpers (for Subject_Data_Segmented format) ─────────────────

def time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_segmented_patient_data(patient_id, subject_data_dir="Subject_Data_Segmented"):
    """Load a single patient's segmented SCG/PPG data and PPG peak annotations.

    Mirrors the relevant parts of HVDNetDataLoader.load_patient_data().
    Returns a dict with keys: signals, fs, signal_length, r_peaks_indices, metadata.
    """
    target_fs = 256

    csv_pattern = os.path.join(subject_data_dir, "**", f"{patient_id}.csv")
    csv_matches = sorted(glob.glob(csv_pattern, recursive=True))
    if not csv_matches:
        raise FileNotFoundError(f"No CSV found for patient {patient_id} in {subject_data_dir}")
    csv_path = csv_matches[0]

    meta_pattern = os.path.join(subject_data_dir, "**", f"{patient_id}_meta.json")
    meta_matches = sorted(glob.glob(meta_pattern, recursive=True))
    meta = {}
    if meta_matches:
        with open(meta_matches[0], "r") as f:
            meta = json.load(f)

    ppg_pattern = os.path.join(subject_data_dir, "**", f"{patient_id}-PPG.json")
    ppg_matches = sorted(glob.glob(ppg_pattern, recursive=True))
    ppg_peak_times = []
    if ppg_matches:
        with open(ppg_matches[0], "r") as f:
            ppg_data = json.load(f)
        ppg_peak_times = ppg_data.get("PPG_Peaks", [])

    df = pd.read_csv(csv_path, comment="#")
    scg_mask = df["x_g"].notna()
    scg_x = df.loc[scg_mask, "x_g"].to_numpy(dtype=np.float32)
    scg_y = df.loc[scg_mask, "y_g"].to_numpy(dtype=np.float32)
    scg_z = df.loc[scg_mask, "z_g"].to_numpy(dtype=np.float32)
    ppg_raw = df.loc[scg_mask, "ppg_raw"].to_numpy(dtype=np.float32)

    original_scg_fs = int(meta.get("sample_rate_scg_hz", target_fs))

    if original_scg_fs != target_fs:
        new_len = int(round(len(scg_x) * target_fs / float(original_scg_fs)))
        scg_x = signal.resample(scg_x, new_len)
        scg_y = signal.resample(scg_y, new_len)
        scg_z = signal.resample(scg_z, new_len)
        ppg_raw = signal.resample(ppg_raw, new_len)

    signal_length = len(scg_x)

    peak_indices = [int(np.round(time_to_seconds(ts) * target_fs)) for ts in ppg_peak_times]
    peak_indices = [idx for idx in peak_indices if 0 <= idx < signal_length]

    return {
        'signals': {'AccX': scg_x, 'AccY': scg_y, 'AccZ': scg_z, 'ECG': ppg_raw},
        'fs': target_fs,
        'signal_length': signal_length,
        'r_peaks_indices': peak_indices,
        'metadata': meta,
    }


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(model_path, device):
    """Load a saved HVDNet model and its metadata.

    Returns (model, metadata_dict) where metadata includes:
      num_classes, d, task_name, class_names, label_columns
    """
    payload = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(payload, dict) and 'model_state_dict' in payload:
        state_dict = payload['model_state_dict']
        num_classes = payload.get('num_classes', 5)
        d_val = payload.get('d', 32)
        task_name = payload.get('task_name', 'Unknown')
        class_names = payload.get('class_names', [str(i) for i in range(num_classes)])
        label_columns = payload.get('label_columns', [])
    else:
        state_dict = payload
        num_classes = 5
        d_val = 32
        task_name = 'Unknown'
        class_names = [str(i) for i in range(num_classes)]
        label_columns = []

    model = HVDNet(num_classes=num_classes, d=d_val)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = {
        'num_classes': num_classes,
        'd': d_val,
        'task_name': task_name,
        'class_names': class_names,
        'label_columns': label_columns,
    }
    return model, metadata


# ── Inference ───────────────────────────────────────────────────────────────

def get_best_torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_tensor_to_device(tensor, device):
    return tensor.to(device, non_blocking=(device.type == "cuda"))


def run_inference_on_patient(model, patient_data, device, is_multilabel=False):
    """Run HVDNet inference on all segments for a single patient.

    Returns a dict with per-segment results and aggregated prediction.
    """
    fs = patient_data['fs']
    signals = patient_data['signals']

    filtered = apply_zero_phase_butterworth(signals, fs)
    r_peaks = np.asarray(patient_data['r_peaks_indices'], dtype=int)
    segments = build_rpeak_segments(r_peaks, patient_data['signal_length'])

    if not segments:
        return {'error': 'No valid segments could be built from the peak annotations'}

    all_logits = []
    segment_results = []

    with torch.no_grad():
        for seg in segments:
            start = seg['start_idx']
            end = seg['end_idx']

            seg_x = pad_or_truncate(zscore_normalize(filtered['AccX'][start:end]), 800)
            seg_y = pad_or_truncate(zscore_normalize(filtered['AccY'][start:end]), 800)
            seg_z = pad_or_truncate(zscore_normalize(filtered['AccZ'][start:end]), 800)

            x_t = torch.tensor(seg_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            y_t = torch.tensor(seg_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            z_t = torch.tensor(seg_z, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            x_t = move_tensor_to_device(x_t, device)
            y_t = move_tensor_to_device(y_t, device)
            z_t = move_tensor_to_device(z_t, device)

            logits, _ = model(x_t, y_t, z_t)
            all_logits.append(logits.cpu())

            segment_results.append({
                'segment_id': seg['segment_id'],
                'start_idx': seg['start_idx'],
                'end_idx': seg['end_idx'],
            })

    all_logits = torch.cat(all_logits, dim=0)

    if is_multilabel:
        probabilities = torch.sigmoid(all_logits).numpy()
        mean_probs = probabilities.mean(axis=0)
        prediction = (mean_probs > 0.5).astype(int)
    else:
        probabilities = torch.softmax(all_logits, dim=1).numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
        vote_counts = np.bincount(predicted_classes, minlength=model.classifier[-1].out_features)
        prediction = int(np.argmax(vote_counts))
        mean_probs = probabilities.mean(axis=0)

    return {
        'num_segments': len(segment_results),
        'segment_predictions': segment_results,
        'mean_probabilities': mean_probs,
        'prediction': prediction,
        'probabilities_all': probabilities,
    }


def find_patients_in_dir(subject_data_dir):
    """Find all patient IDs in Subject_Data_Segmented directory structure."""
    csv_pattern = os.path.join(subject_data_dir, "**", "*.csv")
    paths = sorted(glob.glob(csv_pattern, recursive=True))
    patient_ids = set()
    for p in paths:
        basename = os.path.splitext(os.path.basename(p))[0]
        if not basename.endswith("_meta"):
            patient_ids.add(basename)
    return sorted(patient_ids)


def format_prediction(prediction, class_names, probabilities, is_multilabel):
    if is_multilabel:
        parts = []
        for name, prob in zip(class_names, probabilities):
            parts.append(f"{name}: {prob * 100:.1f}%")
        active = [name for name, p in zip(class_names, prediction) if p > 0.5]
        diag = ", ".join(active) if active else "Normal"
        return f"Diagnosis: {diag} | " + " | ".join(parts)
    else:
        conf = probabilities[prediction] * 100
        probs_str = ", ".join(
            f"{name}: {p * 100:.1f}%"
            for name, p in zip(class_names, probabilities)
        )
        return (
            f"Prediction: {class_names[prediction]} ({conf:.1f}%) | {probs_str}"
        )


# ── Main CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run HVDNet inference on segmented SCG data.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to saved .pt model file",
    )
    parser.add_argument(
        "--patient_id", default=None,
        help="Single patient ID to run inference on",
    )
    parser.add_argument(
        "--subject_dir", default="Subject_Data_Segmented",
        help="Directory containing subject CSV/JSON files (default: Subject_Data_Segmented)",
    )
    parser.add_argument(
        "--multilabel", action="store_true",
        help="Use multi-label (Task III) prediction (sigmoid + threshold 0.5)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all patients found in subject_dir",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.subject_dir):
        print(f"[ERROR] Subject directory not found: {args.subject_dir}")
        sys.exit(1)

    device = get_best_torch_device()
    print(f"[INFO] Using device: {device}")

    model, model_meta = load_model(args.model, device)
    print(f"[INFO] Loaded model: {model_meta['task_name']}")
    print(f"[INFO] Classes: {model_meta['class_names']}")
    print()

    if args.all:
        patient_ids = find_patients_in_dir(args.subject_dir)
        print(f"[INFO] Found {len(patient_ids)} patients in {args.subject_dir}")
        print()
    elif args.patient_id:
        patient_ids = [args.patient_id]
    else:
        print("[ERROR] Specify --patient_id or --all")
        sys.exit(1)

    is_multilabel = args.multilabel or "task iii" in model_meta['task_name'].lower()

    results = []
    for pid in patient_ids:
        try:
            patient_data = load_segmented_patient_data(pid, args.subject_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {pid}: {e}")
            continue

        result = run_inference_on_patient(
            model, patient_data, device, is_multilabel=is_multilabel,
        )

        if 'error' in result:
            print(f"[SKIP] {pid}: {result['error']}")
            continue

        label_str = format_prediction(
            result['prediction'],
            model_meta['class_names'],
            result['mean_probabilities'],
            is_multilabel,
        )
        print(f"[RESULT] {pid} ({result['num_segments']} segments): {label_str}")
        results.append({'patient_id': pid, **result})

    if not results:
        print("[INFO] No results produced.")
        return

    if len(results) > 1:
        print()
        print("─" * 60)
        print("SUMMARY")
        print("─" * 60)
        print(f"Total patients processed: {len(results)}")
        for r in results:
            label_str = format_prediction(
                r['prediction'],
                model_meta['class_names'],
                r['mean_probabilities'],
                is_multilabel,
            )
            print(f"  {r['patient_id']}: {label_str}")


if __name__ == "__main__":
    main()
