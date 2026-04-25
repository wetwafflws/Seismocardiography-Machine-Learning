import glob
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import scipy.ndimage
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class HVDNetDataLoader:
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
        self.label_to_index = {name: idx for idx, name in enumerate(self.label_columns)}
        self.task_class_names = {
            "Task I": ["AS", "MR", "MS", "AR", "N"],
            "Task II": ["AS", "AS-MR", "AS-MS", "AS-AR", "AS-TR"],
            "Task III": ["MS", "MR", "AR", "AS", "TR"],
        }

    def _candidate_paths(self, *parts):
        joined = os.path.join(*parts)
        fallback = "".join(parts)
        if joined == fallback:
            return [joined]
        return [joined, fallback]

    def _resolve_existing_path(self, *parts):
        for path in self._candidate_paths(*parts):
            if os.path.exists(path):
                return path
        return self._candidate_paths(*parts)[0]

    def get_original_fs(self, patient_id):
        if patient_id.startswith("UP-"):
            try:
                num = int(patient_id.split("-")[1])
                if 22 <= num <= 30:
                    return 512
            except ValueError:
                pass
        return 256

    def time_to_seconds(self, time_str):
        h, m, s = time_str.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    def get_task_class_names(self, task_name):
        if task_name not in self.task_class_names:
            raise ValueError("Unknown task: {}".format(task_name))
        return self.task_class_names[task_name]

    def map_label_row_to_index(self, label_row):
        for label_name in self.label_columns:
            if int(label_row.get(label_name, 0)) == 1:
                return self.label_to_index[label_name]
        return None

    def label_row_to_multihot(self, label_row):
        return np.array(
            [float(int(label_row.get(label_name, 0))) for label_name in self.label_columns],
            dtype=np.float32,
        )

    def map_label_row_to_task_index(self, label_row, task_name):
        ms = int(label_row.get("Moderate or greater MS", 0))
        mr = int(label_row.get("Moderate or greater MR", 0))
        ar = int(label_row.get("Moderate or greater AR", 0))
        as_val = int(label_row.get("Moderate or greater AS", 0))
        tr = int(label_row.get("Moderate or greater TR", 0))

        total_positive = ms + mr + ar + as_val + tr

        if task_name == "Task I":
            if total_positive == 0:
                return 4
            if tr == 1:
                return None
            if (as_val + mr + ms + ar) != 1:
                return None
            if as_val == 1:
                return 0
            if mr == 1:
                return 1
            if ms == 1:
                return 2
            if ar == 1:
                return 3
            return None

        if task_name == "Task II":
            if as_val != 1:
                return None
            coexisting_count = mr + ms + ar + tr
            if coexisting_count == 0:
                return 0
            if coexisting_count == 1:
                if mr == 1:
                    return 1
                if ms == 1:
                    return 2
                if ar == 1:
                    return 3
                if tr == 1:
                    return 4
            return None

        if task_name == "Task III":
            return self.label_row_to_multihot(label_row)

        raise ValueError("Unknown task: {}".format(task_name))

    def load_annotation_peaks(self, patient_id, annotation_source, signal_length):
        annotation_source = (annotation_source or "ECG").upper()

        if annotation_source == "AO":
            json_path = self._resolve_existing_path(self.saved_peaks_dir, "{}_AO_Peaks.json".format(patient_id))
            key_name = "{}_AO_Peaks".format(patient_id)
            peak_plot_axis = "AccZ"
            peak_label = "AO-peaks"
        else:
            json_path = self._resolve_existing_path(self.data_dir, "{}-ECG.json".format(patient_id))
            key_name = "LARA_R_Peaks"
            peak_plot_axis = "ECG"
            peak_label = "R-peaks"

        with open(json_path, "r", encoding="utf-8") as f:
            peak_data = json.load(f)

        time_strings = peak_data.get(key_name)
        if time_strings is None:
            time_strings = next(iter(peak_data.values()), [])

        peak_seconds = [self.time_to_seconds(ts) for ts in time_strings]
        peak_indices = [int(np.round(sec * self.target_fs)) for sec in peak_seconds]
        peak_indices = [idx for idx in peak_indices if 0 <= idx < signal_length]

        return {
            "peak_indices": peak_indices,
            "peak_source": annotation_source,
            "peak_plot_axis": peak_plot_axis,
            "peak_label": peak_label,
        }

    def load_labels_table(self):
        labels_path = self._resolve_existing_path(self.data_dir, "ground_truth_labels.csv")
        if not os.path.exists(labels_path):
            return {}

        df_labels = pd.read_csv(labels_path, sep=",")
        df_labels.columns = df_labels.columns.str.strip()
        return {row["Patient ID"]: row.to_dict() for _, row in df_labels.iterrows()}

    def list_available_patients(self):
        csv_paths = sorted(glob.glob(os.path.join(self.data_dir, "Cleaned_*.csv")))
        return [os.path.basename(path).replace("Cleaned_", "").replace(".csv", "") for path in csv_paths]

    def list_eligible_patients_for_task(self, task_name):
        label_lookup = self.load_labels_table()
        patients = self.list_available_patients()
        if task_name == "Task III":
            return patients

        eligible = []
        for patient_id in patients:
            row = label_lookup.get(patient_id)
            if isinstance(row, dict) and self.map_label_row_to_task_index(row, task_name) is not None:
                eligible.append(patient_id)
        return eligible

    def load_patient_data(self, patient_id, annotation_source="ECG"):
        results = {}

        csv_path = self._resolve_existing_path(self.data_dir, "Cleaned_{}.csv".format(patient_id))
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Missing patient CSV: {}".format(csv_path))

        df_signals = pd.read_csv(csv_path, sep=r",")
        original_fs = self.get_original_fs(patient_id)

        scg_x = df_signals["AccX"].values
        scg_y = df_signals["AccY"].values
        scg_z = df_signals["AccZ"].values
        ecg = df_signals["ECG"].values

        if original_fs == 512:
            new_len = len(scg_x) // 2
            scg_x = signal.resample(scg_x, new_len)
            scg_y = signal.resample(scg_y, new_len)
            scg_z = signal.resample(scg_z, new_len)
            ecg = signal.resample(ecg, new_len)

        results["signals"] = {"AccX": scg_x, "AccY": scg_y, "AccZ": scg_z, "ECG": ecg}
        results["fs"] = self.target_fs
        results["signal_length"] = len(scg_x)

        peak_info = self.load_annotation_peaks(patient_id, annotation_source, results["signal_length"])
        results["r_peaks_indices"] = peak_info["peak_indices"]
        results["peak_source"] = peak_info["peak_source"]
        results["peak_plot_axis"] = peak_info["peak_plot_axis"]
        results["peak_label"] = peak_info["peak_label"]

        label_lookup = self.load_labels_table()
        label_row = label_lookup.get(patient_id)
        if isinstance(label_row, dict):
            results["labels"] = label_row
            results["label_vector"] = self.label_row_to_multihot(label_row)
            results["label_index"] = self.map_label_row_to_index(label_row)
        else:
            results["labels"] = None
            results["label_vector"] = None
            results["label_index"] = None

        return results


class sCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.skip_projection = nn.Identity()
        if in_channels != out_channels:
            self.skip_projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip_projection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        out = self.pool(out)

        return out


class SCNN_Module(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, kernel_size=7):
        super().__init__()
        channels = (base_filters, base_filters // 2, base_filters // 4)
        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(sCNN_Block(c_in, c_out, kernel_size))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


sCNN_Module = SCNN_Module


class LSTM_Module(nn.Module):
    def __init__(self, input_features=16, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
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
    def __init__(self, num_classes=5, d=64):
        super().__init__()

        self.scnn_x = sCNN_Module(in_channels=1, base_filters=d)
        self.scnn_y = sCNN_Module(in_channels=1, base_filters=d)
        self.scnn_z = sCNN_Module(in_channels=1, base_filters=d)

        self.lstm_x = LSTM_Module(input_features=d // 4, hidden_size=d)
        self.lstm_y = LSTM_Module(input_features=d // 4, hidden_size=d)
        self.lstm_z = LSTM_Module(input_features=d // 4, hidden_size=d)

        self.sa_x = SA_Module(hidden_size=d)
        self.sa_y = SA_Module(hidden_size=d)
        self.sa_z = SA_Module(hidden_size=d)

        self.classifier = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.BatchNorm1d(d),
            nn.Dropout(p=0.2),
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

        concat_vector = torch.cat((ctx_x, ctx_y, ctx_z), dim=1)
        logits = self.classifier(concat_vector)

        return logits, (attn_x, attn_y, attn_z)


class InferenceEngine:
    def __init__(self, loader):
        self.loader = loader

    @staticmethod
    def zscore_normalize(values):
        values = np.asarray(values, dtype=float)
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val < 1e-12:
            return np.zeros_like(values)
        return (values - mean_val) / std_val

    @staticmethod
    def pad_or_truncate(values, target_len=800):
        values = np.asarray(values, dtype=float)
        if len(values) < target_len:
            return np.pad(values, (0, target_len - len(values)), mode="constant")
        if len(values) > target_len:
            return values[:target_len]
        return values

    @staticmethod
    def apply_zero_phase_butterworth(signals_dict, fs, lowcut=1.0, highcut=30.0, order=6):
        nyquist = fs / 2.0
        if not (0 < lowcut < highcut < nyquist):
            raise ValueError("Invalid bandpass range: {}-{} Hz for fs={}".format(lowcut, highcut, fs))

        b, a = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
        filtered = {}
        for name, values in signals_dict.items():
            filtered[name] = signal.filtfilt(b, a, np.asarray(values, dtype=float))
        return filtered

    @staticmethod
    def build_rpeak_segments(r_peaks, signal_length):
        segments = []
        for i in range(len(r_peaks) - 3):
            start_idx = int(r_peaks[i])
            end_idx = int(r_peaks[i + 3])
            if 0 <= start_idx < end_idx <= signal_length:
                segments.append({
                    "segment_id": i,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_peak_number": i,
                    "end_peak_number": i + 3,
                })
        return segments

    def build_attention_sample_from_segment(self, filtered_signals, segment):
        start_idx = segment["start_idx"]
        end_idx = segment["end_idx"]

        seg_x = self.pad_or_truncate(self.zscore_normalize(filtered_signals["AccX"][start_idx:end_idx]), 800)
        seg_y = self.pad_or_truncate(self.zscore_normalize(filtered_signals["AccY"][start_idx:end_idx]), 800)
        seg_z = self.pad_or_truncate(self.zscore_normalize(filtered_signals["AccZ"][start_idx:end_idx]), 800)

        x_tensor = torch.tensor(seg_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y_tensor = torch.tensor(seg_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        z_tensor = torch.tensor(seg_z, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return x_tensor, y_tensor, z_tensor

    @staticmethod
    def resize_attention_to_length(attention_values, target_len):
        attention_values = np.asarray(attention_values, dtype=float).reshape(-1)
        if target_len <= 0:
            return np.asarray([], dtype=float)
        if len(attention_values) == 0:
            return np.zeros(target_len, dtype=float)

        scale = target_len / max(len(attention_values), 1)
        resized = scipy.ndimage.zoom(attention_values, scale, order=1)
        if len(resized) < target_len:
            resized = np.pad(resized, (0, target_len - len(resized)), mode="edge")
        return resized[:target_len]

    def build_patient_inference_result(self, patient_id, task_name, model, device, annotation_source):
        data = self.loader.load_patient_data(patient_id, annotation_source=annotation_source)
        data["filtered_signals"] = self.apply_zero_phase_butterworth(
            data["signals"],
            data["fs"],
            lowcut=1.0,
            highcut=30.0,
            order=6,
        )
        data["segments"] = self.build_rpeak_segments(data["r_peaks_indices"], data["signal_length"])

        if not data["segments"]:
            raise RuntimeError("No segments available for patient {}".format(patient_id))

        if task_name == "Task III" and data.get("label_vector") is None:
            raise RuntimeError("Missing Task III label vector for patient {}".format(patient_id))

        segment_results = []
        signal_length = int(data["signal_length"])
        full_attention_sum = {
            "X": np.zeros(signal_length, dtype=float),
            "Y": np.zeros(signal_length, dtype=float),
            "Z": np.zeros(signal_length, dtype=float),
        }
        full_attention_count = {
            "X": np.zeros(signal_length, dtype=float),
            "Y": np.zeros(signal_length, dtype=float),
            "Z": np.zeros(signal_length, dtype=float),
        }

        model.eval()
        with torch.no_grad():
            for segment_idx, segment in enumerate(data["segments"]):
                x_tensor, y_tensor, z_tensor = self.build_attention_sample_from_segment(data["filtered_signals"], segment)
                logits, (attn_x, attn_y, attn_z) = model(x_tensor.to(device), y_tensor.to(device), z_tensor.to(device))

                segment_probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                if task_name == "Task III":
                    segment_prediction = (segment_probabilities > 0.5).astype(float)
                    true_label = np.asarray(data["label_vector"], dtype=float)
                else:
                    segment_prediction = int(torch.argmax(logits, dim=1).item())
                    if isinstance(data.get("labels"), dict):
                        true_label = self.loader.map_label_row_to_task_index(data["labels"], task_name)
                    else:
                        true_label = None

                attn_arrays = {
                    "X": attn_x.squeeze().detach().cpu().numpy(),
                    "Y": attn_y.squeeze().detach().cpu().numpy(),
                    "Z": attn_z.squeeze().detach().cpu().numpy(),
                }

                segment_length = int(segment["end_idx"] - segment["start_idx"])
                for axis_name, attention_values in attn_arrays.items():
                    resized_attention = self.resize_attention_to_length(attention_values, segment_length)
                    start_idx = int(segment["start_idx"])
                    end_idx = int(segment["end_idx"])
                    full_attention_sum[axis_name][start_idx:end_idx] += resized_attention
                    full_attention_count[axis_name][start_idx:end_idx] += 1.0

                segment_results.append(
                    {
                        "segment_idx": segment_idx,
                        "segment": segment,
                        "start_idx": int(segment["start_idx"]),
                        "end_idx": int(segment["end_idx"]),
                        "segment_length": segment_length,
                        "probabilities": segment_probabilities,
                        "prediction": segment_prediction,
                        "true_label": true_label,
                    }
                )

        full_attention = {}
        for axis_name in ("X", "Y", "Z"):
            full_attention[axis_name] = np.divide(
                full_attention_sum[axis_name],
                full_attention_count[axis_name],
                out=np.zeros_like(full_attention_sum[axis_name]),
                where=full_attention_count[axis_name] > 0,
            )

        segment_probability_matrix = np.stack([item["probabilities"] for item in segment_results], axis=0)
        mean_probabilities = np.mean(segment_probability_matrix, axis=0)

        if task_name == "Task III":
            final_prediction = (mean_probabilities > 0.5).astype(float)
            actual_truth = np.asarray(data["label_vector"], dtype=float)
        else:
            vote_counts = np.bincount(
                np.asarray([int(item["prediction"]) for item in segment_results], dtype=int),
                minlength=len(self.loader.get_task_class_names(task_name)),
            )
            final_prediction = int(np.argmax(vote_counts))
            if isinstance(data.get("labels"), dict):
                actual_truth = self.loader.map_label_row_to_task_index(data["labels"], task_name)
            else:
                actual_truth = None

        is_exact_match = False
        if actual_truth is not None:
            if task_name == "Task III":
                is_exact_match = bool(np.array_equal(final_prediction, actual_truth))
            else:
                is_exact_match = int(final_prediction) == int(actual_truth)

        return {
            "patient_id": patient_id,
            "task_name": task_name,
            "data": data,
            "segment_results": segment_results,
            "full_attention": full_attention,
            "mean_probabilities": mean_probabilities,
            "final_prediction": final_prediction,
            "actual_truth": actual_truth,
            "is_exact_match": is_exact_match,
            "class_names": self.loader.get_task_class_names(task_name),
            "num_segments": len(segment_results),
        }


class BatchInferenceWorker(QThread):
    progress_update = pyqtSignal(int, int, object)
    error_update = pyqtSignal(str)
    finished_update = pyqtSignal(list)

    def __init__(self, engine, model, device, task_name, annotation_source, patient_ids, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.model = model
        self.device = device
        self.task_name = task_name
        self.annotation_source = annotation_source
        self.patient_ids = list(patient_ids)
        self._should_stop = False

    def stop(self):
        self._should_stop = True

    def run(self):
        results = []
        total = len(self.patient_ids)
        for idx, patient_id in enumerate(self.patient_ids, start=1):
            if self._should_stop:
                break
            try:
                result = self.engine.build_patient_inference_result(
                    patient_id,
                    self.task_name,
                    self.model,
                    self.device,
                    self.annotation_source,
                )
                results.append(result)
                self.progress_update.emit(idx, total, result)
            except Exception as exc:
                self.error_update.emit("{}: {}".format(patient_id, exc))

        self.finished_update.emit(results)


class InferenceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVDNet Inference Viewer (PyQt5)")
        self.resize(1500, 900)

        self.loader = HVDNetDataLoader(data_dir="Data")
        self.engine = InferenceEngine(self.loader)

        self.model = None
        self.device = torch.device("cpu")
        self.loaded_model_task = None
        self.batch_worker = None
        self.last_result = None

        self._build_ui()
        self._set_default_model_for_task()
        self.refresh_patients()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)

        controls = QGridLayout()
        row = 0

        controls.addWidget(QLabel("Data directory"), row, 0)
        self.data_dir_edit = QLineEdit("Data")
        controls.addWidget(self.data_dir_edit, row, 1)
        data_btn = QPushButton("Browse")
        data_btn.clicked.connect(self.choose_data_dir)
        controls.addWidget(data_btn, row, 2)
        row += 1

        controls.addWidget(QLabel("Task"), row, 0)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Task I", "Task II", "Task III"])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        controls.addWidget(self.task_combo, row, 1)

        controls.addWidget(QLabel("Annotation"), row, 2)
        self.annotation_combo = QComboBox()
        self.annotation_combo.addItems(["ECG", "AO"])
        controls.addWidget(self.annotation_combo, row, 3)
        row += 1

        controls.addWidget(QLabel("Model checkpoint"), row, 0)
        self.model_path_edit = QLineEdit()
        controls.addWidget(self.model_path_edit, row, 1, 1, 2)
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.choose_model)
        controls.addWidget(model_browse_btn, row, 3)
        row += 1

        self.load_model_btn = QPushButton("Load model")
        self.load_model_btn.clicked.connect(self.load_model)
        controls.addWidget(self.load_model_btn, row, 0)

        controls.addWidget(QLabel("Patient"), row, 1)
        self.patient_combo = QComboBox()
        self.patient_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        controls.addWidget(self.patient_combo, row, 2)

        refresh_btn = QPushButton("Refresh patients")
        refresh_btn.clicked.connect(self.refresh_patients)
        controls.addWidget(refresh_btn, row, 3)
        row += 1

        self.run_single_btn = QPushButton("Run single inference")
        self.run_single_btn.clicked.connect(self.run_single_inference)
        controls.addWidget(self.run_single_btn, row, 0)

        self.run_batch_btn = QPushButton("Run batch")
        self.run_batch_btn.clicked.connect(self.run_batch_inference)
        controls.addWidget(self.run_batch_btn, row, 1)

        self.stop_batch_btn = QPushButton("Stop batch")
        self.stop_batch_btn.clicked.connect(self.stop_batch_inference)
        self.stop_batch_btn.setEnabled(False)
        controls.addWidget(self.stop_batch_btn, row, 2)

        self.save_plot_btn = QPushButton("Save current plot")
        self.save_plot_btn.clicked.connect(self.save_current_plot)
        controls.addWidget(self.save_plot_btn, row, 3)
        row += 1

        self.show_attention_check = QCheckBox("Show attention overlay")
        self.show_attention_check.setChecked(True)
        self.show_attention_check.stateChanged.connect(self.refresh_plot_from_last_result)
        controls.addWidget(self.show_attention_check, row, 0, 1, 2)

        self.export_batch_check = QCheckBox("Export batch CSV and PNG")
        self.export_batch_check.setChecked(True)
        controls.addWidget(self.export_batch_check, row, 2, 1, 2)
        row += 1

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        controls.addWidget(self.progress_bar, row, 0, 1, 4)
        row += 1

        root.addLayout(controls)

        self.figure = Figure(figsize=(12, 9))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setMinimumHeight(460)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, stretch=3)

        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(["Patient", "Prediction", "Truth", "Match", "Segments"])
        root.addWidget(self.batch_table, stretch=1)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        root.addWidget(self.log_text, stretch=1)

    def log(self, message):
        self.log_text.append(message)

    def choose_data_dir(self):
        selected = QFileDialog.getExistingDirectory(self, "Select Data directory", self.data_dir_edit.text().strip() or ".")
        if selected:
            self.data_dir_edit.setText(selected)
            self.loader.data_dir = selected
            self.refresh_patients()

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select model checkpoint", "", "PyTorch Model (*.pt *.pth)")
        if path:
            self.model_path_edit.setText(path)

    def on_task_changed(self, _task_name):
        self._set_default_model_for_task()
        self.refresh_patients()

    def _set_default_model_for_task(self):
        task_name = self.task_combo.currentText()
        default_map = {
            "Task I": "hvdnet_task_i_1.pt",
            "Task II": "hvdnet_task_ii_1.pt",
            "Task III": "hvdnet_task_iii_1.pt",
        }
        default_model = default_map.get(task_name)
        if default_model and os.path.exists(default_model):
            self.model_path_edit.setText(default_model)

    def refresh_patients(self):
        self.loader.data_dir = self.data_dir_edit.text().strip() or "Data"
        task_name = self.task_combo.currentText()
        patient_ids = self.loader.list_eligible_patients_for_task(task_name)

        current = self.patient_combo.currentData()
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        for patient_id in patient_ids:
            self.patient_combo.addItem(patient_id, patient_id)
        self.patient_combo.blockSignals(False)

        if current:
            idx = self.patient_combo.findData(current)
            if idx >= 0:
                self.patient_combo.setCurrentIndex(idx)

        if self.patient_combo.count() == 0:
            self.log("No eligible patients found for {} in {}".format(task_name, self.loader.data_dir))
        else:
            self.log("Loaded {} eligible patients for {}".format(self.patient_combo.count(), task_name))

    def load_model(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Missing model", "Select a checkpoint file first.")
            return
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Missing model", "Checkpoint not found: {}".format(model_path))
            return

        try:
            payload = torch.load(model_path, map_location="cpu")
            if isinstance(payload, dict) and "model_state_dict" in payload:
                state_dict = payload["model_state_dict"]
                task_name = payload.get("task_name", self.task_combo.currentText())
                class_names = payload.get("class_names", self.loader.get_task_class_names(task_name))
            else:
                state_dict = payload
                task_name = self.task_combo.currentText()
                class_names = self.loader.get_task_class_names(task_name)

            model = HVDNet(num_classes=5, d=64)
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            self.loaded_model_task = task_name
            self.device = torch.device("cpu")
            self.model.to(self.device)

            if task_name in ["Task I", "Task II", "Task III"] and task_name != self.task_combo.currentText():
                self.task_combo.setCurrentText(task_name)

            self.log("Loaded model: {}".format(model_path))
            self.log("Task: {} | Classes: {}".format(task_name, class_names))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.log("Model load failed: {}".format(exc))

    def ensure_ready_for_inference(self):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return False
        return True

    def get_active_task_name(self):
        if self.loaded_model_task in ["Task I", "Task II", "Task III"]:
            return self.loaded_model_task
        return self.task_combo.currentText()

    def get_selected_patient_id(self):
        patient_id = self.patient_combo.currentData()
        return str(patient_id).strip() if patient_id else ""

    def run_single_inference(self):
        if not self.ensure_ready_for_inference():
            return

        patient_id = self.get_selected_patient_id()
        if not patient_id:
            QMessageBox.warning(self, "No patient", "Select a patient first.")
            return

        task_name = self.get_active_task_name()
        annotation_source = self.annotation_combo.currentText()

        try:
            result = self.engine.build_patient_inference_result(
                patient_id,
                task_name,
                self.model,
                self.device,
                annotation_source,
            )
            self.last_result = result
            self.render_result(result)
            self.log(self.format_result_text(result))
        except Exception as exc:
            self.log("Single inference failed: {}".format(exc))
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Inference failed", str(exc))

    def run_batch_inference(self):
        if not self.ensure_ready_for_inference():
            return

        task_name = self.get_active_task_name()
        patient_ids = [self.patient_combo.itemData(i) for i in range(self.patient_combo.count()) if self.patient_combo.itemData(i)]
        if not patient_ids:
            QMessageBox.warning(self, "No patients", "No eligible patients available for batch run.")
            return

        self.batch_table.setRowCount(0)
        self.progress_bar.setValue(0)

        self.batch_worker = BatchInferenceWorker(
            engine=self.engine,
            model=self.model,
            device=self.device,
            task_name=task_name,
            annotation_source=self.annotation_combo.currentText(),
            patient_ids=patient_ids,
            parent=self,
        )
        self.batch_worker.progress_update.connect(self.on_batch_progress)
        self.batch_worker.error_update.connect(self.on_batch_error)
        self.batch_worker.finished_update.connect(self.on_batch_finished)

        self.run_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.batch_worker.start()
        self.log("Batch started for {} patients".format(len(patient_ids)))

    def stop_batch_inference(self):
        if self.batch_worker is not None:
            self.batch_worker.stop()
            self.log("Stop requested for batch run")

    def on_batch_progress(self, current, total, result):
        percent = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(percent)
        self.append_batch_row(result)
        self.last_result = result
        self.render_result(result)
        self.log("Batch progress: {}/{} | {}".format(current, total, result["patient_id"]))

    def on_batch_error(self, message):
        self.log("Batch item failed: {}".format(message))

    def on_batch_finished(self, results):
        self.run_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.progress_bar.setValue(100 if results else 0)
        self.log("Batch completed. Successful patients: {}".format(len(results)))

        if self.export_batch_check.isChecked() and results:
            self.export_batch_outputs(results)

    def append_batch_row(self, result):
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)

        prediction_text, truth_text = self.format_prediction_truth(result)
        match_text = "YES" if result["is_exact_match"] else "NO"

        self.batch_table.setItem(row, 0, QTableWidgetItem(result["patient_id"]))
        self.batch_table.setItem(row, 1, QTableWidgetItem(prediction_text))
        self.batch_table.setItem(row, 2, QTableWidgetItem(truth_text))
        self.batch_table.setItem(row, 3, QTableWidgetItem(match_text))
        self.batch_table.setItem(row, 4, QTableWidgetItem(str(result["num_segments"])))

    def format_prediction_truth(self, result):
        class_names = result["class_names"]
        task_name = result["task_name"]

        if task_name == "Task III":
            pred_labels = [name for name, val in zip(class_names, result["final_prediction"]) if val > 0.5]
            prediction_text = ", ".join(pred_labels) if pred_labels else "Normal"

            truth = result.get("actual_truth")
            if truth is None:
                truth_text = "Unknown"
            else:
                truth_labels = [name for name, val in zip(class_names, truth) if val > 0.5]
                truth_text = ", ".join(truth_labels) if truth_labels else "Normal"
        else:
            prediction_text = class_names[int(result["final_prediction"])]
            truth = result.get("actual_truth")
            truth_text = class_names[int(truth)] if truth is not None else "Unknown"

        return prediction_text, truth_text

    def format_result_text(self, result):
        prediction_text, truth_text = self.format_prediction_truth(result)
        mean_prob = ", ".join(
            ["{}={:.1f}%".format(name, p * 100.0) for name, p in zip(result["class_names"], result["mean_probabilities"])]
        )
        return (
            "Patient={} | Task={} | Segments={} | Pred={} | Truth={} | Match={} | MeanProbs=[{}]".format(
                result["patient_id"],
                result["task_name"],
                result["num_segments"],
                prediction_text,
                truth_text,
                "YES" if result["is_exact_match"] else "NO",
                mean_prob,
            )
        )

    def refresh_plot_from_last_result(self):
        if self.last_result is not None:
            self.render_result(self.last_result)

    def render_result(self, result):
        data = result["data"]
        fs = data["fs"]
        signals = data["filtered_signals"]
        n = data["signal_length"]
        t = np.arange(n) / fs

        self.figure.clear()
        ax1 = self.figure.add_subplot(4, 1, 1)
        ax2 = self.figure.add_subplot(4, 1, 2, sharex=ax1)
        ax3 = self.figure.add_subplot(4, 1, 3, sharex=ax1)
        ax4 = self.figure.add_subplot(4, 1, 4, sharex=ax1)

        self._plot_axis(ax1, t, signals["AccX"], "AccX", result["full_attention"]["X"])
        self._plot_axis(ax2, t, signals["AccY"], "AccY", result["full_attention"]["Y"])
        self._plot_axis(ax3, t, signals["AccZ"], "AccZ", result["full_attention"]["Z"])

        ax4.plot(t, signals["ECG"], color="#d62728", linewidth=0.8)
        ax4.set_ylabel("ECG")
        ax4.grid(True, alpha=0.25)

        peak_indices = np.asarray(data.get("r_peaks_indices", []), dtype=int)
        peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(t))]
        if len(peak_indices):
            peak_times = peak_indices / fs
            peak_axis = ax3 if data.get("peak_plot_axis") == "AccZ" else ax4
            for peak_t in peak_times:
                peak_axis.axvline(x=float(peak_t), color="gold", alpha=0.2, linewidth=0.8)

        prediction_text, truth_text = self.format_prediction_truth(result)
        title = "{} | {} | Pred={} | Truth={}".format(result["patient_id"], result["task_name"], prediction_text, truth_text)
        ax1.set_title(title)
        ax4.set_xlabel("Time (s)")
        self.figure.subplots_adjust(left=0.065, right=0.99, top=0.92, bottom=0.12, hspace=0.75)
        self.canvas.draw_idle()

    def _plot_axis(self, axis, t, values, axis_name, attention_values):
        values = np.asarray(values, dtype=float)
        axis.plot(t, values, linewidth=0.8)
        axis.set_ylabel(axis_name)
        axis.grid(True, alpha=0.25)

        if not self.show_attention_check.isChecked():
            return

        attn = np.asarray(attention_values, dtype=float)
        if len(attn) != len(values):
            return

        y_min = float(np.min(values) - 0.1 * (np.max(values) - np.min(values) + 1e-6))
        y_max = float(np.max(values) + 0.1 * (np.max(values) - np.min(values) + 1e-6))

        heat = np.expand_dims(attn, axis=0)
        axis.imshow(
            heat,
            extent=[t[0] if len(t) else 0.0, t[-1] if len(t) else 1.0, y_min, y_max],
            aspect="auto",
            cmap="viridis",
            alpha=0.35,
            origin="lower",
        )

    def export_batch_outputs(self, results):
        out_dir = os.path.join(os.getcwd(), "inference_outputs")
        os.makedirs(out_dir, exist_ok=True)

        summary_rows = []
        for result in results:
            prediction_text, truth_text = self.format_prediction_truth(result)
            summary_rows.append(
                {
                    "patient_id": result["patient_id"],
                    "task": result["task_name"],
                    "prediction": prediction_text,
                    "truth": truth_text,
                    "match": int(bool(result["is_exact_match"])),
                    "num_segments": result["num_segments"],
                }
            )

            self.render_result(result)
            png_path = os.path.join(out_dir, "{}_{}.png".format(result["task_name"].replace(" ", "_"), result["patient_id"]))
            self.figure.savefig(png_path, dpi=120)

        csv_path = os.path.join(out_dir, "batch_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        self.log("Exported batch summary: {}".format(csv_path))

    def save_current_plot(self):
        if self.last_result is None:
            QMessageBox.information(self, "No plot", "Run inference first.")
            return

        default_name = "{}_{}.png".format(self.last_result["task_name"].replace(" ", "_"), self.last_result["patient_id"])
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", default_name, "PNG (*.png)")
        if path:
            self.figure.savefig(path, dpi=120)
            self.log("Saved plot: {}".format(path))


def main():
    app = QApplication(sys.argv)
    window = InferenceWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
