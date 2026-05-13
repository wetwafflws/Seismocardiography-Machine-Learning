"""
XGBoost Binary Classifier for Heart Valve Disease
Based on: Yang et al. (2020) - Scientific Reports

Task: Binary classification — AS vs Normal (N)
Validation: Leave-One-Subject-Out (LOSO)
Features: CWT statistical features (MEAN, MEDIAN, MAX, STD, IQR) per frequency bin
Feature selection: Elastic Net
Classifier: XGBoost

Data loading follows the exact same pipeline as machinelearning.py (HVDNet project).

Usage:
    python xgboost_classifier.py --data_dir /path/to/Data/ --annotation ECG
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

import pywt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# DATA LOADING (mirrors HVDNetDataLoader exactly)
# ─────────────────────────────────────────────

class DataLoader:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.saved_peaks_dir = "Saved_Peaks"
        self.target_fs = 256
        self.label_columns = [
            "Moderate or greater MS",
            "Moderate or greater MR",
            "Moderate or greater AR",
            "Moderate or greater AS",
            "Moderate or greater TR",
        ]

    def get_original_fs(self, patient_id):
        if patient_id.startswith('UP-'):
            try:
                num = int(patient_id.split('-')[1])
                if 22 <= num <= 30:
                    return 512
            except ValueError:
                pass
        return 256

    def time_to_seconds(self, time_str):
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def load_annotation_peaks(self, patient_id, annotation_source, signal_length):
        annotation_source = (annotation_source or "ECG").upper()

        if annotation_source == "AO":
            json_path = os.path.join(self.saved_peaks_dir, f"{patient_id}_AO_Peaks.json")
            key_name = f"{patient_id}_AO_Peaks"
        else:
            json_path = f"{self.data_dir}{patient_id}-ECG.json"
            key_name = 'LARA_R_Peaks'

        with open(json_path, 'r') as f:
            peak_data = json.load(f)

        time_strings = peak_data.get(key_name)
        if time_strings is None:
            if peak_data:
                time_strings = next(iter(peak_data.values()))
            else:
                time_strings = []

        peak_seconds = [self.time_to_seconds(ts) for ts in time_strings]
        peak_indices = [int(np.round(sec * self.target_fs)) for sec in peak_seconds]
        peak_indices = [idx for idx in peak_indices if 0 <= idx < signal_length]

        return peak_indices

    def load_patient_data(self, patient_id, annotation_source="ECG"):
        csv_path = f"{self.data_dir}Cleaned_{patient_id}.csv"
        df = pd.read_csv(csv_path, sep=',')
        original_fs = self.get_original_fs(patient_id)

        scg_x = df['AccX'].values
        scg_y = df['AccY'].values
        scg_z = df['AccZ'].values
        ecg   = df['ECG'].values

        if original_fs == 512:
            new_len = len(scg_x) // 2
            scg_x = scipy_signal.resample(scg_x, new_len)
            scg_y = scipy_signal.resample(scg_y, new_len)
            scg_z = scipy_signal.resample(scg_z, new_len)
            ecg   = scipy_signal.resample(ecg, new_len)

        signal_length = len(scg_x)
        r_peaks = self.load_annotation_peaks(patient_id, annotation_source, signal_length)

        labels_path = f"{self.data_dir}ground_truth_labels.csv"
        df_labels = pd.read_csv(labels_path, sep=',')
        df_labels.columns = df_labels.columns.str.strip()
        patient_row = df_labels[df_labels['Patient ID'] == patient_id]

        if patient_row.empty:
            raise ValueError(f"No label found for patient {patient_id}")

        label_dict = patient_row.iloc[0].to_dict()

        return {
            'signals': {'AccX': scg_x, 'AccY': scg_y, 'AccZ': scg_z, 'ECG': ecg},
            'fs': self.target_fs,
            'signal_length': signal_length,
            'r_peaks_indices': r_peaks,
            'labels': label_dict,
        }

    def get_binary_label(self, label_dict):
        """
        Returns binary label for AS vs N classification.
            0 = AS  (only AS positive, no other conditions)
            1 = N   (all conditions negative)
            None = skip (co-existing conditions or TR-only)
        """
        ms     = int(label_dict.get("Moderate or greater MS", 0))
        mr     = int(label_dict.get("Moderate or greater MR", 0))
        ar     = int(label_dict.get("Moderate or greater AR", 0))
        as_val = int(label_dict.get("Moderate or greater AS", 0))
        tr     = int(label_dict.get("Moderate or greater TR", 0))
        total  = ms + mr + ar + as_val + tr

        if total == 0:
            return 1  # Normal
        if as_val == 1 and total == 1:
            return 0  # Pure AS only
        return None  # Co-existing or other — skip for binary task


# ─────────────────────────────────────────────
# PREPROCESSING (mirrors HVDNet exactly)
# ─────────────────────────────────────────────

def apply_butterworth(signals_dict, fs, lowcut=1.0, highcut=30.0, order=6):
    b, a = scipy_signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    filtered = {}
    for key, arr in signals_dict.items():
        filtered[key] = scipy_signal.filtfilt(b, a, arr)
    return filtered


def build_rpeak_segments(r_peaks, signal_length):
    segments = []
    for i in range(len(r_peaks) - 3):
        start_idx = int(r_peaks[i])
        end_idx   = int(r_peaks[i + 3])
        if 0 <= start_idx < end_idx <= signal_length:
            segments.append({'start_idx': start_idx, 'end_idx': end_idx})
    return segments


def zscore_normalize(arr):
    mu = np.mean(arr)
    sd = np.std(arr)
    if sd < 1e-8:
        return arr - mu
    return (arr - mu) / sd


def pad_or_truncate(arr, target_len=800):
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode='constant')


# ─────────────────────────────────────────────
# FEATURE EXTRACTION (CWT statistical features)
# Following Yang et al. 2020: MEAN, MED, MAX, STD, IQR
# per frequency bin across all 3 axes
# ─────────────────────────────────────────────

def extract_cwt_features(seg_x, seg_y, seg_z, fs=256, n_freqs=60,
                          freq_min=1.0, freq_max=30.0):
    """
    Extract CWT statistical features from a single 3-axis SCG segment.

    Yang et al. use Morse wavelet (not available in PyWavelets).
    We use Complex Morlet (cmor1.5-1.0) which is the standard substitute.

    Returns:
        1D numpy array of shape (n_freqs * 5 stats * 3 axes,) = (60*5*3,) = 900 features
    """
    wavelet = 'cmor1.5-1.0'
    central_freq = pywt.central_frequency(wavelet)

    freqs  = np.linspace(freq_min, freq_max, n_freqs)
    scales = central_freq * fs / freqs

    all_features = []

    for seg in [seg_x, seg_y, seg_z]:
        coeffs, _ = pywt.cwt(seg, scales, wavelet, sampling_period=1.0 / fs)
        power = np.abs(coeffs)  # shape: (n_freqs, n_time)

        feat_mean   = np.mean(power, axis=1)
        feat_median = np.median(power, axis=1)
        feat_max    = np.max(power, axis=1)
        feat_std    = np.std(power, axis=1)
        feat_iqr    = (np.percentile(power, 75, axis=1)
                       - np.percentile(power, 25, axis=1))

        axis_feats = np.concatenate([
            feat_mean, feat_median, feat_max, feat_std, feat_iqr
        ])
        all_features.append(axis_feats)

    return np.concatenate(all_features)  # 900-dim vector


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────

def build_dataset(data_dir, annotation_source="ECG"):
    """
    Load all patients, extract CWT features per segment.

    Returns:
        patient_features : list of np.array (n_segments, 900) — one per patient
        patient_labels   : list of int (0=AS, 1=N) — one per patient
        patient_ids      : list of str
    """
    loader = DataLoader(data_dir=data_dir)

    csv_paths = sorted(glob.glob(os.path.join(data_dir, "Cleaned_*.csv")))
    if not csv_paths:
        raise RuntimeError(f"No Cleaned_*.csv files found in {data_dir}")

    patient_features = []
    patient_labels   = []
    patient_ids      = []

    for csv_path in csv_paths:
        patient_id = os.path.basename(csv_path).replace("Cleaned_", "").replace(".csv", "")

        try:
            data = loader.load_patient_data(patient_id, annotation_source)

            binary_label = loader.get_binary_label(data['labels'])
            if binary_label is None:
                print(f"  [SKIP] {patient_id} — co-existing or unsupported condition")
                continue

            filtered = apply_butterworth(data['signals'], data['fs'])
            segments = build_rpeak_segments(data['r_peaks_indices'], data['signal_length'])

            if not segments:
                print(f"  [SKIP] {patient_id} — no valid segments")
                continue

            seg_features = []
            for seg in segments:
                s, e = seg['start_idx'], seg['end_idx']

                seg_x = pad_or_truncate(zscore_normalize(filtered['AccX'][s:e]))
                seg_y = pad_or_truncate(zscore_normalize(filtered['AccY'][s:e]))
                seg_z = pad_or_truncate(zscore_normalize(filtered['AccZ'][s:e]))

                feats = extract_cwt_features(seg_x, seg_y, seg_z)
                seg_features.append(feats)

            label_name = "AS" if binary_label == 0 else "N"
            print(f"  [OK] {patient_id} | Class: {label_name} | Segments: {len(seg_features)}")

            patient_features.append(np.array(seg_features))
            patient_labels.append(binary_label)
            patient_ids.append(patient_id)

        except Exception as ex:
            print(f"  [ERROR] {patient_id}: {ex}")
            continue

    return patient_features, patient_labels, patient_ids


# ─────────────────────────────────────────────
# LOSO TRAINING AND EVALUATION
# ─────────────────────────────────────────────

def run_loso(patient_features, patient_labels, patient_ids):
    """
    Leave-One-Subject-Out cross-validation with:
      - StandardScaler
      - ElasticNet feature selection
      - XGBoost classification
      - Majority vote across segments for patient-level prediction
    """
    n_patients = len(patient_ids)
    print(f"\n{'='*60}")
    print(f"LOSO Validation | {n_patients} patients")
    print(f"{'='*60}")

    all_true = []
    all_pred = []

    for i in range(n_patients):
        test_id    = patient_ids[i]
        test_label = patient_labels[i]
        X_test     = patient_features[i]

        # Build train set from all other patients
        train_X_list = []
        train_y_list = []
        for j in range(n_patients):
            if j == i:
                continue
            n_segs = len(patient_features[j])
            train_X_list.append(patient_features[j])
            train_y_list.extend([patient_labels[j]] * n_segs)

        X_train = np.vstack(train_X_list)
        y_train = np.array(train_y_list)

        # Standardize
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Feature selection (L1-based, significantly faster than ElasticNet with saga)
        en = LogisticRegressionCV(
            penalty='l1',
            solver='liblinear',
            cv=3,
            max_iter=2000,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            # Mute the scikit-learn 1.10 FutureWarning
        )
        # Suppress the legacy attributes warning for LogisticRegressionCV
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            en.fit(X_train_sc, y_train)
        mask = en.coef_[0] != 0

        if mask.sum() == 0:
            mask = np.ones(X_train_sc.shape[1], dtype=bool)  # fallback

        X_train_sel = X_train_sc[:, mask]
        X_test_sel  = X_test_sc[:, mask]

        # Calculate scale_pos_weight for XGBoost to handle class imbalance
        count_0 = np.sum(y_train == 0)
        count_1 = np.sum(y_train == 1)
        scale_weight = count_0 / count_1 if count_1 > 0 else 1.0

        # XGBoost
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            scale_pos_weight=scale_weight
        )
        clf.fit(X_train_sel, y_train)

        # Segment-level predictions → majority vote → patient prediction
        seg_preds    = clf.predict(X_test_sel)
        patient_pred = int(np.bincount(seg_preds).argmax())

        true_name = "AS" if test_label == 0 else "N"
        pred_name = "AS" if patient_pred == 0 else "N"
        correct   = "✓" if patient_pred == test_label else "✗"

        print(f"  {correct} Patient {test_id:10s} | True: {true_name} | Pred: {pred_name} "
              f"| Features selected: {mask.sum()} / {len(mask)}")

        all_true.append(test_label)
        all_pred.append(patient_pred)

    return all_true, all_pred


# ─────────────────────────────────────────────
# RESULTS REPORTING
# ─────────────────────────────────────────────

def report_results(all_true, all_pred):
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    acc   = accuracy_score(all_true, all_pred)
    prec  = precision_score(all_true, all_pred, average='macro', zero_division=0)
    rec   = recall_score(all_true, all_pred, average='macro', zero_division=0)
    f1    = f1_score(all_true, all_pred, average='macro', zero_division=0)
    cm    = confusion_matrix(all_true, all_pred)

    print(f"\nOverall Accuracy : {acc*100:.2f}%")
    print(f"Macro Precision  : {prec*100:.2f}%")
    print(f"Macro Recall     : {rec*100:.2f}%")
    print(f"Macro F1         : {f1*100:.2f}%")

    print(f"\nConfusion Matrix:")
    print(f"              Pred AS   Pred N")
    print(f"True AS   {cm[0,0]:8d} {cm[0,1]:8d}")
    print(f"True N    {cm[1,0]:8d} {cm[1,1]:8d}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<8} {'Sensitivity':>12} {'Specificity':>12} {'Precision':>10} {'F1':>8}")
    print("-" * 56)

    class_names = ['AS', 'N']
    for idx, name in enumerate(class_names):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens  = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec_ = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

        print(f"{name:<8} {sens*100:>11.2f}% {spec*100:>11.2f}% "
              f"{prec_*100:>9.2f}% {f1_*100:>7.2f}%")

    print(f"\nComparison with Yang et al. (2020) XGBoost:")
    print(f"  Paper accuracy (LOSO): 93%")
    print(f"  Your accuracy (LOSO):  {acc*100:.2f}%")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost binary AS classifier — Yang et al. 2020"
    )
    parser.add_argument(
        '--data_dir', type=str, default='Data/',
        help='Path to data directory containing Cleaned_*.csv and ground_truth_labels.csv'
    )
    parser.add_argument(
        '--annotation', type=str, default='ECG',
        choices=['ECG', 'AO'],
        help='Annotation source for R-peak detection'
    )
    args = parser.parse_args()

    print("XGBoost Binary Classifier — Yang et al. (2020)")
    print(f"Data directory : {args.data_dir}")
    print(f"Annotation     : {args.annotation}")
    print(f"Task           : Binary AS vs N (Leave-One-Subject-Out)")
    print()

    # 1. Build dataset
    print("Loading patients and extracting CWT features...")
    print("(This may take several minutes — CWT is computed per segment)\n")
    patient_features, patient_labels, patient_ids = build_dataset(
        args.data_dir, args.annotation
    )

    n_as = patient_labels.count(0)
    n_n  = patient_labels.count(1)
    print(f"\nDataset summary: {len(patient_ids)} patients | AS={n_as} | N={n_n}")

    if len(patient_ids) < 3:
        print("[ERROR] Not enough patients for LOSO. Need at least 3.")
        sys.exit(1)

    # 2. LOSO
    all_true, all_pred = run_loso(patient_features, patient_labels, patient_ids)

    # 3. Report
    report_results(all_true, all_pred)


if __name__ == "__main__":
    main()