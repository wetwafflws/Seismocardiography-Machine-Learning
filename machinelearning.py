import sys
import json
import os
import glob
import threading
import time
import numpy as np
import pandas as pd
from scipy import signal
# Keeps execution on MPS when possible and falls back only unsupported ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import scipy.ndimage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QComboBox, QTabWidget, QGridLayout, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF

# Ensure pyqtgraph uses the same Qt binding as the app.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import pyqtgraph as pg


def get_best_torch_device():
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_tensor_to_device(tensor, device):
    """Use non_blocking transfer only where it is supported."""
    return tensor.to(device, non_blocking=(device.type == "cuda"))

class HVDNetDataLoader:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.saved_peaks_dir = "Saved_Peaks"
        self.target_fs = 256  # The paper standardizes to 256 Hz 
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

    def get_original_fs(self, patient_id):
        """Determine original sampling frequency based on database rules."""
        if patient_id.startswith('UP-'):
            try:
                num = int(patient_id.split('-')[1])
                if 22 <= num <= 30:
                    return 512
            except ValueError:
                pass
        return 256  # Default for CP-01 to CP-70 and UP-01 to UP-21

    def time_to_seconds(self, time_str):
        """Converts HH:MM:SS.ssss to total seconds."""
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def load_annotation_peaks(self, patient_id, annotation_source, signal_length):
        annotation_source = (annotation_source or "ECG").upper()

        if annotation_source == "AO":
            json_path = os.path.join(self.saved_peaks_dir, f"{patient_id}_AO_Peaks.json")
            key_name = f"{patient_id}_AO_Peaks"
            peak_plot_axis = 'AccZ'
            peak_label = 'AO-peaks'
        else:
            json_path = f"{self.data_dir}{patient_id}-ECG.json"
            key_name = 'LARA_R_Peaks'
            peak_plot_axis = 'ECG'
            peak_label = 'R-peaks'

        try:
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

            return {
                'peak_indices': peak_indices,
                'peak_source': annotation_source,
                'peak_plot_axis': peak_plot_axis,
                'peak_label': peak_label,
            }
        except Exception as e:
            raise Exception(f"Failed to load {annotation_source} peaks JSON: {str(e)}")

    def load_patient_data(self, patient_id, annotation_source="ECG"):
        """Loads SCG, ECG, selected peaks annotation, and Ground Truth for a given patient."""
        results = {}
        
        # 1. Load Raw Signals (CSV)
        # Using \s+ as separator since Excel copy-paste usually yields tabs or multiple spaces
        csv_path = f"{self.data_dir}Cleaned_{patient_id}.csv"
        try:
            df_signals = pd.read_csv(csv_path, sep=r',')
            original_fs = self.get_original_fs(patient_id)
            
            scg_x = df_signals['AccX'].values
            scg_y = df_signals['AccY'].values
            scg_z = df_signals['AccZ'].values
            ecg = df_signals['ECG'].values

            # Resample to 256 Hz if necessary 
            if original_fs == 512:
                # Calculate new length for 256 Hz (exactly half)
                new_len = len(scg_x) // 2
                scg_x = signal.resample(scg_x, new_len)
                scg_y = signal.resample(scg_y, new_len)
                scg_z = signal.resample(scg_z, new_len)
                ecg = signal.resample(ecg, new_len)
                
            results['signals'] = {
                'AccX': scg_x,
                'AccY': scg_y,
                'AccZ': scg_z,
                'ECG': ecg
            }
            results['fs'] = self.target_fs
            results['signal_length'] = len(scg_x)
            
        except Exception as e:
            raise Exception(f"Failed to load signal CSV: {str(e)}")

        # 2. Load selected annotation peaks (ECG R-peaks or AO peaks)
        peak_info = self.load_annotation_peaks(patient_id, annotation_source, results['signal_length'])
        results['r_peaks_indices'] = peak_info['peak_indices']
        results['peak_source'] = peak_info['peak_source']
        results['peak_plot_axis'] = peak_info['peak_plot_axis']
        results['peak_label'] = peak_info['peak_label']

        # 3. Load Ground Truth Labels
        labels_path = f"{self.data_dir}ground_truth_labels.csv"
        try:
            # Assuming tab separated if copied from excel. Adjust sep if it's comma-separated.
            df_labels = pd.read_csv(labels_path, sep=',') 
            
            # Standardize column names to remove leading/trailing whitespaces
            df_labels.columns = df_labels.columns.str.strip()
            
            patient_row = df_labels[df_labels['Patient ID'] == patient_id]
            if not patient_row.empty:
                label_row = patient_row.iloc[0]
                label_dict = label_row.to_dict()
                positive_labels = [
                    label_name for label_name in self.label_columns
                    if int(label_dict.get(label_name, 0)) == 1
                ]

                results['labels'] = label_dict
                results['positive_labels'] = positive_labels
                results['label_vector'] = self.label_row_to_multihot(label_dict)
                results['label_index'] = self.map_label_row_to_index(label_dict)
                results['label_name'] = self.index_to_label_name(results['label_index'])
            else:
                results['labels'] = "No labels found for this patient."
                results['positive_labels'] = []
                results['label_vector'] = None
                results['label_index'] = None
                results['label_name'] = None
                
        except Exception as e:
            raise Exception(f"Failed to load Ground Truth Labels: {str(e)}")

        return results

    def map_label_row_to_index(self, label_row):
        """Map a label row to the first positive valve label for Task I/II compatibility."""
        for label_name in self.label_columns:
            if int(label_row.get(label_name, 0)) == 1:
                return self.label_to_index[label_name]
        return None

    def label_row_to_multihot(self, label_row):
        """Convert the CSV valve annotations into a 5-element multi-hot vector."""
        return np.array([
            float(int(label_row.get(label_name, 0)))
            for label_name in self.label_columns
        ], dtype=np.float32)

    def index_to_label_name(self, label_index):
        if label_index is None:
            return None
        return self.label_columns[label_index]

    def get_task_class_names(self, task_name):
        if task_name not in self.task_class_names:
            raise ValueError(f"Unknown task: {task_name}")
        return self.task_class_names[task_name]

    def map_label_row_to_task_index(self, label_row, task_name):
        """Map raw multi-label valve annotations into task-specific class indices."""
        ms = int(label_row.get("Moderate or greater MS", 0))
        mr = int(label_row.get("Moderate or greater MR", 0))
        ar = int(label_row.get("Moderate or greater AR", 0))
        as_val = int(label_row.get("Moderate or greater AS", 0))
        tr = int(label_row.get("Moderate or greater TR", 0))

        total_positive = ms + mr + ar + as_val + tr

        if task_name == "Task I":
            # Exact classes: AS, MR, MS, AR, N (no co-existing diseases)
            if total_positive == 0:
                return 4  # N
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
            # Exact classes: AS, AS-MR, AS-MS, AS-AR, AS-TR
            if as_val != 1:
                return None
            coexisting_count = mr + ms + ar + tr
            if coexisting_count == 0:
                return 0  # AS
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

        raise ValueError(f"Unknown task: {task_name}")


class sCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(sCNN_Block, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
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


# Paper naming compatibility alias.
sCNN_Module = SCNN_Module


class LSTM_Module(nn.Module):
    def __init__(self, input_features=16, hidden_size=64):
        super(LSTM_Module, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # Convert (batch, channels, seq_len) -> (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out


class SA_Module(nn.Module):
    def __init__(self, hidden_size=64):
        super(SA_Module, self).__init__()
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores = torch.tanh(self.dense(lstm_out))
        weights = F.softmax(scores, dim=1)
        weighted_out = lstm_out * weights
        context_vector = torch.sum(weighted_out, dim=1)
        return context_vector, weights


class HVDNet(nn.Module):
    def __init__(self, num_classes=5, d=64):
        super(HVDNet, self).__init__()

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
            nn.Linear(d, num_classes)
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


class TrainingWorker(QThread):
    epoch_update = pyqtSignal(int, int, float, float, float, float, float, float)
    log_update = pyqtSignal(str)
    test_update = pyqtSignal(float, float)
    finished_update = pyqtSignal(object)
    error_update = pyqtSignal(str)

    def __init__(self, x_tensor, y_tensor, z_tensor, label_tensor, num_classes=5, d=64,
                 num_epochs=100, batch_size=64, learning_rate=0.001, weight_decay=0.004,
                 test_size=0.2, n_splits=5, random_state=42, multi_label=False, parent=None):
        super().__init__(parent)
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.z_tensor = z_tensor
        self.label_tensor = label_tensor
        self.num_classes = num_classes
        self.d = d
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.multi_label = multi_label
        self._pause_event = threading.Event()
        self._pause_event.set()

    def pause_training(self):
        self._pause_event.clear()

    def resume_training(self):
        self._pause_event.set()

    def wait_if_paused(self):
        self._pause_event.wait()

    def compute_class_weights(self, labels_np):
        class_counts = np.bincount(labels_np, minlength=self.num_classes)
        total = class_counts.sum()
        weights = np.zeros(self.num_classes, dtype=np.float32)
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total / (self.num_classes * count)
            else:
                weights[i] = 0.0
        return torch.tensor(weights, dtype=torch.float32), class_counts

    def evaluate_loader(self, model, criterion, loader, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_z, labels in loader:
                batch_x = move_tensor_to_device(batch_x, device)
                batch_y = move_tensor_to_device(batch_y, device)
                batch_z = move_tensor_to_device(batch_z, device)
                labels = move_tensor_to_device(labels, device)
                if self.multi_label:
                    labels = labels.float()

                logits, _ = model(batch_x, batch_y, batch_z)
                loss = criterion(logits, labels)

                batch_weight = labels.numel() if self.multi_label else labels.size(0)
                running_loss += loss.item() * batch_weight
                if self.multi_label:
                    pred = (logits > 0.0).float()
                    total += labels.numel()
                    correct += (pred == labels).float().sum().item()
                else:
                    pred = torch.argmax(logits, dim=1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

        return running_loss / max(total, 1), 100.0 * correct / max(total, 1)

    def run(self):
        try:
            device = get_best_torch_device()
            self.log_update.emit(f"Training on device: {device}")
            self.log_update.emit(f"Epochs: {self.num_epochs}, Batch Size: {self.batch_size}")
            total_epoch_steps = self.n_splits * self.num_epochs
            completed_epoch_steps = 0
            epoch_durations = []

            labels_np = self.label_tensor.cpu().numpy()
            all_indices = np.arange(len(labels_np))

            if self.multi_label:
                if len(labels_np) < 2:
                    raise RuntimeError("Need at least two samples for train/test split.")
                train_idx, test_idx = train_test_split(
                    all_indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=True,
                )
                train_labels = labels_np[train_idx]
                if len(train_idx) < self.n_splits:
                    raise RuntimeError(
                        f"Need at least {self.n_splits} samples for KFold cross validation."
                    )
                splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            else:
                unique_classes, class_freq = np.unique(labels_np, return_counts=True)
                if len(unique_classes) < 2:
                    raise RuntimeError("Need at least two classes for stratified train/test split.")
                if np.min(class_freq) < 2:
                    raise RuntimeError("At least one class has fewer than 2 samples; cannot perform stratified 80/20 split.")

                train_idx, test_idx = train_test_split(
                    all_indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=labels_np,
                )

                train_labels = labels_np[train_idx]
                unique_train, train_freq = np.unique(train_labels, return_counts=True)
                if np.min(train_freq) < self.n_splits:
                    raise RuntimeError(
                        f"Need at least {self.n_splits} samples per class in training split for StratifiedKFold. "
                        f"Minimum class count is {np.min(train_freq)}."
                    )
                splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            self.log_update.emit(f"80/20 split done: train={len(train_idx)} samples, test={len(test_idx)} samples")

            x_test = self.x_tensor[test_idx]
            y_test = self.y_tensor[test_idx]
            z_test = self.z_tensor[test_idx]
            label_test = self.label_tensor[test_idx]
            test_loader = DataLoader(
                TensorDataset(x_test, y_test, z_test, label_test),
                batch_size=self.batch_size,
                shuffle=False,
            )

            # Initialize model once — weights carry over across folds
            model = HVDNet(num_classes=self.num_classes, d=self.d).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            best_val_loss = float('inf')
            best_fold = -1
            best_state_dict = None

            for fold_idx, (fold_train_rel, fold_val_rel) in enumerate(splitter.split(train_idx, train_labels), start=1):
                self.wait_if_paused()
                fold_train_idx = train_idx[fold_train_rel]
                fold_val_idx = train_idx[fold_val_rel]

                x_train = self.x_tensor[fold_train_idx]
                y_train = self.y_tensor[fold_train_idx]
                z_train = self.z_tensor[fold_train_idx]
                label_train = self.label_tensor[fold_train_idx]

                x_val = self.x_tensor[fold_val_idx]
                y_val = self.y_tensor[fold_val_idx]
                z_val = self.z_tensor[fold_val_idx]
                label_val = self.label_tensor[fold_val_idx]

                train_loader = DataLoader(
                    TensorDataset(x_train, y_train, z_train, label_train),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                val_loader = DataLoader(
                    TensorDataset(x_val, y_val, z_val, label_val),
                    batch_size=self.batch_size,
                    shuffle=False,
                )

                if self.multi_label:
                    fold_counts = label_train.sum(dim=0).cpu().numpy()
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    fold_class_weights, fold_counts = self.compute_class_weights(label_train.cpu().numpy())
                    criterion = nn.CrossEntropyLoss(weight=fold_class_weights.to(device))

                # Reset only the classifier head for each new fold
                def reset_linear(m):
                    if isinstance(m, nn.Linear):
                        m.reset_parameters()
                model.classifier.apply(reset_linear)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.num_epochs, eta_min=1e-6
                )

                self.log_update.emit(
                    f"Fold {fold_idx}/{self.n_splits} | train={len(fold_train_idx)} val={len(fold_val_idx)} "
                    f"| train class counts={fold_counts.tolist()}"
                )

                for epoch in range(self.num_epochs):
                    self.wait_if_paused()
                    epoch_start = time.perf_counter()
                    model.train()
                    running_loss = 0.0
                    correct_predictions = 0
                    total_samples = 0

                    for batch_x, batch_y, batch_z, labels in train_loader:
                        self.wait_if_paused()
                        batch_x = move_tensor_to_device(batch_x, device)
                        batch_y = move_tensor_to_device(batch_y, device)
                        batch_z = move_tensor_to_device(batch_z, device)
                        labels = move_tensor_to_device(labels, device)
                        if self.multi_label:
                            labels = labels.float()

                        optimizer.zero_grad()
                        logits, _ = model(batch_x, batch_y, batch_z)
                        loss = criterion(logits, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        running_loss += loss.item() * labels.size(0)
                        if self.multi_label:
                            predicted = (logits > 0.0).float()
                            total_samples += labels.numel()
                            correct_predictions += (predicted == labels).float().sum().item()
                        else:
                            predicted = torch.argmax(logits, dim=1)
                            total_samples += labels.size(0)
                            correct_predictions += (predicted == labels).sum().item()

                    train_loss = running_loss / max(total_samples, 1)
                    train_acc = 100.0 * correct_predictions / max(total_samples, 1)
                    val_loss, val_acc = self.evaluate_loader(model, criterion, val_loader, device)
                    scheduler.step()
                    epoch_seconds = time.perf_counter() - epoch_start

                    completed_epoch_steps += 1
                    epoch_durations.append(epoch_seconds)
                    avg_epoch_seconds = float(np.mean(epoch_durations))
                    eta_seconds = avg_epoch_seconds * max(total_epoch_steps - completed_epoch_steps, 0)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_fold = fold_idx
                        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                    self.epoch_update.emit(
                        fold_idx,
                        epoch + 1,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        epoch_seconds,
                        eta_seconds,
                    )

            if best_state_dict is None:
                raise RuntimeError("Cross-validation did not produce a valid best model.")

            model.load_state_dict(best_state_dict)
            if self.multi_label:
                test_criterion = nn.BCEWithLogitsLoss()
            else:
                test_criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = self.evaluate_loader(model, test_criterion, test_loader, device)
            self.test_update.emit(test_loss, test_acc)

            self.finished_update.emit({
                'best_state_dict': best_state_dict,
                'best_fold': best_fold,
                'best_val_loss': best_val_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'train_size': int(len(train_idx)),
                'test_size': int(len(test_idx)),
            })
        except Exception as exc:
            self.error_update.emit(str(exc))


class HVDMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVDNet - Step-by-Step Preprocessing")
        self.resize(1000, 800)

        pg.setConfigOptions(antialias=False)

        self.current_patient_id = None
        self.current_data = None
        self.segments = []
        self.preprocessed_segments = []
        self.current_segment_idx = 0
        self.scnn_models = {}
        self.lstm_models = {}
        self.sa_models = {}
        self.hvdnet_model = None
        self.training_worker = None
        self.train_loader = None
        self.class_weights = None
        self.current_training_task = "Task I"
        self.train_epochs = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_steps = []
        self.last_training_summary = None
        self.loaded_model_task = None
        self.loaded_model_classes = None
        self.dataset_class_summary_cache = None
        self.last_inference_result = None
        self.is_training_paused = False
        self.current_num_epochs = 100
        self.current_n_splits = 5
        
        self.loader = HVDNetDataLoader(data_dir="Data/") # Assumes scripts and data are in the same folder
        
        # UI Setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QWidget()
        left_panel_container_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(240)
        left_panel.setMaximumWidth(420)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        controls_container = QWidget()
        left_layout = QVBoxLayout(controls_container)
        controls_scroll.setWidget(controls_container)
        left_panel_container_layout.addWidget(controls_scroll)

        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Three columns: controls | text output | plots
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(middle_panel, 2)
        main_layout.addWidget(right_panel, 7)
        
        # Controls panel (vertical stack)
        input_layout = QVBoxLayout()
        input_layout.setSpacing(6)
        self.patient_input = QComboBox()
        self.patient_input.setMinimumWidth(180)
        self.patient_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.patient_input.setToolTip("Select a patient from the available cleaned files")
        self.task_selector = QComboBox()
        self.task_selector.addItems(["Task I", "Task II", "Task III"])
        self.task_selector.currentTextChanged.connect(self.on_task_changed)
        self.annotation_selector = QComboBox()
        self.annotation_selector.addItems(["ECG R-peaks", "AO peaks (Saved_Peaks)"])
        self.peak_filter_selector = QComboBox()
        self.peak_filter_selector.addItems(["Keep all peaks", "Discard impossible peaks"])
        self.peak_filter_selector.setToolTip("Drops peaks with inter-peak intervals outside 0.50s to 1.50s (40-120 BPM) before segmentation when enabled.")
        self.step1_btn = QPushButton("Step 1: Load Data")
        self.step1_btn.clicked.connect(self.step_load_data)
        self.step2_btn = QPushButton("Step 2:\nFilter (1-30 Hz)")
        self.step2_btn.clicked.connect(self.step_apply_filter)
        self.step3_btn = QPushButton("Step 3:\nBuild Segments")
        self.step3_btn.clicked.connect(self.step_build_segments)
        self.step4_btn = QPushButton("Step 4:\nZ-Score + Pad/Truncate 800")
        self.step4_btn.clicked.connect(self.step_preprocess_segments)
        self.step5_btn = QPushButton("Step 5:\nBuild sCNN + Forward Check")
        self.step5_btn.clicked.connect(self.step_build_cnn)
        self.step6_btn = QPushButton("Step 6:\nBuild LSTM + Forward Check")
        self.step6_btn.clicked.connect(self.step_build_lstm)
        self.step7_btn = QPushButton("Step 7:\nBuild SA + Forward Check")
        self.step7_btn.clicked.connect(self.step_build_sa)
        self.step8_btn = QPushButton("Step 8:\nBuild HVDNet + Final Check")
        self.step8_btn.clicked.connect(self.step_build_hvdnet)
        self.step9_btn = QPushButton("Step 9:\nTrain Model")
        self.step9_btn.clicked.connect(self.step_train_model)
        self.small_training_mode = QComboBox()
        self.small_training_mode.addItems(["Full training (5-fold, 100 epochs)", "Small training (3-fold, 50 epochs)"])
        self.small_training_mode.setToolTip("Use fewer folds and epochs for quicker iteration.")
        self.pause_resume_btn = QPushButton("Pause Training")
        self.pause_resume_btn.clicked.connect(self.step_toggle_training_pause)
        self.load_model_btn = QPushButton("Load Saved\nModel")
        self.load_model_btn.clicked.connect(self.step_load_saved_model)
        self.infer_btn = QPushButton("Run Inference\nAttention")
        self.infer_btn.clicked.connect(self.step_run_inference_attention)
        self.inference_view_mode = QComboBox()
        self.inference_view_mode.addItems(["Whole Signal", "Segment Heatmap"])
        self.inference_view_mode.currentIndexChanged.connect(self.on_inference_view_mode_changed)
        self.cm_btn = QPushButton("Generate\nConfusion Matrix")
        self.cm_btn.clicked.connect(self.step_generate_confusion_matrix)
        self.class_attn_class_selector = QComboBox()
        self.class_attn_class_selector.setMinimumWidth(120)
        self.class_attn_class_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.class_attn_samples_input = QLineEdit("50")
        self.class_attn_samples_input.setMinimumWidth(60)
        self.class_attn_samples_input.setMaximumWidth(72)
        self.class_attn_samples_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.class_attn_btn = QPushButton("Class Mean\nAttention")
        self.class_attn_btn.clicked.connect(self.step_generate_class_attention_maps)
        self.save_model_btn = QPushButton("Save Trained\nModel")
        self.save_model_btn.clicked.connect(self.step_save_model)

        patient_row = QHBoxLayout()
        patient_row.addWidget(QLabel("Patient ID:"))
        patient_row.addWidget(self.patient_input)

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task:"))
        task_row.addWidget(self.task_selector)

        annotation_row = QHBoxLayout()
        annotation_row.addWidget(QLabel("Annotations:"))
        annotation_row.addWidget(self.annotation_selector)

        peak_filter_row = QHBoxLayout()
        peak_filter_row.addWidget(QLabel("Peak Filter:"))
        peak_filter_row.addWidget(self.peak_filter_selector)

        class_attn_row = QHBoxLayout()
        class_attn_row.addWidget(QLabel("Condition:"))
        class_attn_row.addWidget(self.class_attn_class_selector)
        class_attn_row.addWidget(QLabel("n:"))
        class_attn_row.addWidget(self.class_attn_samples_input)
        class_attn_row.addStretch()

        all_control_buttons = [
            self.step1_btn, self.step2_btn, self.step3_btn, self.step4_btn, self.step5_btn,
            self.step6_btn, self.step7_btn, self.step8_btn, self.step9_btn, self.pause_resume_btn, self.load_model_btn,
            self.infer_btn, self.cm_btn, self.class_attn_btn, self.save_model_btn,
            self.prev_segment_btn if hasattr(self, 'prev_segment_btn') else None,
            self.next_segment_btn if hasattr(self, 'next_segment_btn') else None,
        ]
        for btn in all_control_buttons:
            if btn is None:
                continue
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.setMinimumHeight(44)

        input_layout.addLayout(patient_row)
        input_layout.addLayout(task_row)
        input_layout.addLayout(annotation_row)
        input_layout.addLayout(peak_filter_row)
        input_layout.addWidget(self.step1_btn)
        input_layout.addWidget(self.step2_btn)
        input_layout.addWidget(self.step3_btn)
        input_layout.addWidget(self.step4_btn)
        input_layout.addWidget(self.step5_btn)
        input_layout.addWidget(self.step6_btn)
        input_layout.addWidget(self.step7_btn)
        input_layout.addWidget(self.step8_btn)
        input_layout.addWidget(self.small_training_mode)
        input_layout.addWidget(self.step9_btn)
        input_layout.addWidget(self.pause_resume_btn)
        input_layout.addWidget(self.load_model_btn)
        input_layout.addWidget(self.infer_btn)
        inference_view_row = QHBoxLayout()
        inference_view_row.addWidget(QLabel("Inference View:"))
        inference_view_row.addWidget(self.inference_view_mode)
        input_layout.addLayout(inference_view_row)
        input_layout.addWidget(self.cm_btn)
        input_layout.addLayout(class_attn_row)
        input_layout.addWidget(self.class_attn_btn)
        input_layout.addWidget(self.save_model_btn)
        
        left_layout.addLayout(input_layout)

        self.populate_patient_dropdown(self.task_selector.currentText())

        # View mode and segment navigation
        nav_layout = QVBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Entire Signal", "Segments (P_i to P_{i+3})"])
        self.view_mode.currentIndexChanged.connect(self.on_view_mode_changed)

        self.plot_stage = QComboBox()
        self.plot_stage.addItems(["Raw", "Filtered", "Preprocessed Segments"])
        self.plot_stage.currentIndexChanged.connect(self.plot_current_view)

        self.prev_segment_btn = QPushButton("Previous Segment")
        self.prev_segment_btn.clicked.connect(self.show_previous_segment)
        self.prev_segment_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_segment_btn.setMinimumHeight(36)

        self.next_segment_btn = QPushButton("Next Segment")
        self.next_segment_btn.clicked.connect(self.show_next_segment)
        self.next_segment_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_segment_btn.setMinimumHeight(36)

        self.segment_info_label = QLabel("Segment: N/A")
        self.segment_info_label.setWordWrap(True)
        self.segment_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        view_row.addWidget(self.view_mode)

        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Plot Stage:"))
        stage_row.addWidget(self.plot_stage)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.prev_segment_btn)
        btn_row.addWidget(self.next_segment_btn)

        info_row = QHBoxLayout()
        info_row.addWidget(self.segment_info_label)

        nav_layout.addLayout(view_row)
        nav_layout.addLayout(stage_row)
        nav_layout.addLayout(btn_row)
        nav_layout.addLayout(info_row)
        left_layout.addLayout(nav_layout)
        
        # Console output (middle narrow column)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        middle_layout.addWidget(self.console)

        # Tabbed graph area
        self.graph_tabs = QTabWidget()
        right_layout.addWidget(self.graph_tabs)

        self.signals_tab = QWidget()
        self.signals_tab_layout = QVBoxLayout(self.signals_tab)
        self.training_tab = QWidget()
        self.training_tab_layout = QVBoxLayout(self.training_tab)
        self.confusion_tab = QWidget()
        self.confusion_tab_layout = QVBoxLayout(self.confusion_tab)
        self.class_attention_tab = QWidget()
        self.class_attention_tab_layout = QVBoxLayout(self.class_attention_tab)
        self.inference_tab = QWidget()
        self.inference_tab_layout = QVBoxLayout(self.inference_tab)

        self.graph_tabs.addTab(self.signals_tab, "Signals")
        self.graph_tabs.addTab(self.training_tab, "Training")
        self.graph_tabs.addTab(self.confusion_tab, "Confusion Matrix")
        self.graph_tabs.addTab(self.class_attention_tab, "Class Attention")
        self.graph_tabs.addTab(self.inference_tab, "Inference")

        # Live training diagnostics
        self.train_diag_widget = pg.GraphicsLayoutWidget(show=False)
        self.train_diag_widget.setMinimumHeight(220)
        self.training_tab_layout.addWidget(self.train_diag_widget)

        self.loss_plot = self.train_diag_widget.addPlot(row=0, col=0)
        self.loss_plot.setTitle("Training Loss")
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setLabel('bottom', 'Step (fold-epoch)')
        self.loss_plot.showGrid(x=True, y=True, alpha=0.25)
        self.loss_train_curve = self.loss_plot.plot([], [], pen=pg.mkPen('#ff7f0e', width=2), name='Train Loss')
        self.loss_val_curve = self.loss_plot.plot([], [], pen=pg.mkPen('#1f77b4', width=2), name='Val Loss')

        self.acc_plot = self.train_diag_widget.addPlot(row=1, col=0)
        self.acc_plot.setTitle("Training Accuracy")
        self.acc_plot.setLabel('left', 'Accuracy (%)')
        self.acc_plot.setLabel('bottom', 'Step (fold-epoch)')
        self.acc_plot.showGrid(x=True, y=True, alpha=0.25)
        self.acc_train_curve = self.acc_plot.plot([], [], pen=pg.mkPen('#2ca02c', width=2), name='Train Acc')
        self.acc_val_curve = self.acc_plot.plot([], [], pen=pg.mkPen('#d62728', width=2), name='Val Acc')

        # Confusion matrix view
        self.cm_widget = pg.GraphicsLayoutWidget(show=False)
        self.cm_widget.setMinimumHeight(240)
        self.confusion_tab_layout.addWidget(self.cm_widget)
        self.cm_plot = self.cm_widget.addPlot(row=0, col=0)
        self.cm_plot.setTitle("Confusion Matrix (Held-out Test Set)")
        self.cm_plot.setLabel('left', 'True Class')
        self.cm_plot.setLabel('bottom', 'Predicted Class')
        self.cm_plot.showGrid(x=False, y=False)

        # Class mean attention heatmaps
        self.class_attention_widget = pg.GraphicsLayoutWidget(show=False)
        self.class_attention_tab_layout.addWidget(self.class_attention_widget)

        # Inference view area
        self.inference_widget = pg.GraphicsLayoutWidget(show=False)
        self.inference_tab_layout.addWidget(self.inference_widget)
        self.inf_accx_plot = self.inference_widget.addPlot(row=0, col=0)
        self.inf_accy_plot = self.inference_widget.addPlot(row=1, col=0)
        self.inf_accz_plot = self.inference_widget.addPlot(row=2, col=0)
        self.inf_ecg_plot = self.inference_widget.addPlot(row=3, col=0)

        self.inf_accy_plot.setXLink(self.inf_accx_plot)
        self.inf_accz_plot.setXLink(self.inf_accx_plot)
        self.inf_ecg_plot.setXLink(self.inf_accx_plot)

        self.inf_accx_plot.setLabel('left', 'AccX')
        self.inf_accy_plot.setLabel('left', 'AccY')
        self.inf_accz_plot.setLabel('left', 'AccZ')
        self.inf_ecg_plot.setLabel('left', 'ECG')
        self.inf_ecg_plot.setLabel('bottom', 'Samples')

        for p in (self.inf_accx_plot, self.inf_accy_plot, self.inf_accz_plot, self.inf_ecg_plot):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setClipToView(True)
            p.setDownsampling(mode='peak')

        # Interactive signal plots (PyQtGraph)
        self.graph_widget = pg.GraphicsLayoutWidget(show=False)
        self.signals_tab_layout.addWidget(self.graph_widget)

        self.accx_plot = self.graph_widget.addPlot(row=0, col=0)
        self.accy_plot = self.graph_widget.addPlot(row=1, col=0)
        self.accz_plot = self.graph_widget.addPlot(row=2, col=0)
        self.ecg_plot = self.graph_widget.addPlot(row=3, col=0)

        self.accy_plot.setXLink(self.accx_plot)
        self.accz_plot.setXLink(self.accx_plot)
        self.ecg_plot.setXLink(self.accx_plot)

        self.accx_plot.setLabel('left', 'AccX')
        self.accy_plot.setLabel('left', 'AccY')
        self.accz_plot.setLabel('left', 'AccZ')
        self.ecg_plot.setLabel('left', 'ECG')
        self.ecg_plot.setLabel('bottom', 'Time (s)')

        for p in (self.accx_plot, self.accy_plot, self.accz_plot, self.ecg_plot):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setClipToView(True)
            p.setDownsampling(mode='peak')

        self.refresh_class_attention_class_selector()
        self.update_step_controls()
        self.update_navigation_controls()

    def log(self, message):
        self.console.append(message)

    def refresh_class_attention_class_selector(self):
        task_name = self.task_selector.currentText()
        class_names = self.loader.get_task_class_names(task_name)

        current = self.class_attn_class_selector.currentText()
        self.class_attn_class_selector.blockSignals(True)
        self.class_attn_class_selector.clear()
        self.class_attn_class_selector.addItems(class_names)
        if current in class_names:
            self.class_attn_class_selector.setCurrentText(current)
        self.class_attn_class_selector.blockSignals(False)

    def on_inference_view_mode_changed(self, _index):
        if self.last_inference_result is not None:
            self.render_last_inference()
        self.update_navigation_controls()

    def on_task_changed(self, task_name):
        self.refresh_class_attention_class_selector()
        self.populate_patient_dropdown(task_name)

    def format_patient_condition_text(self, label_row):
        if not isinstance(label_row, dict):
            return "No labels"

        positive_labels = [
            label_name for label_name in self.loader.label_columns
            if int(label_row.get(label_name, 0)) == 1
        ]
        return ", ".join(positive_labels) if positive_labels else "Normal"

    def is_patient_eligible_for_task(self, label_row, task_name):
        if not isinstance(label_row, dict):
            return False

        return self.loader.map_label_row_to_task_index(label_row, task_name) is not None

    def populate_patient_dropdown(self, task_name=None):
        if not hasattr(self, 'patient_input'):
            return

        task_name = task_name or self.task_selector.currentText()

        csv_paths = sorted(glob.glob(os.path.join(self.loader.data_dir, "Cleaned_*.csv")))
        labels_path = os.path.join(self.loader.data_dir, "ground_truth_labels.csv")
        label_lookup = {}

        if os.path.exists(labels_path):
            df_labels = pd.read_csv(labels_path, sep=',')
            df_labels.columns = df_labels.columns.str.strip()
            label_lookup = {
                row['Patient ID']: row.to_dict()
                for _, row in df_labels.iterrows()
            }

        current_patient_id = self.current_patient_id
        current_model_data = self.patient_input.currentData() if self.patient_input.count() > 0 else None

        self.patient_input.blockSignals(True)
        self.patient_input.clear()

        for csv_path in csv_paths:
            patient_id = os.path.basename(csv_path).replace("Cleaned_", "").replace(".csv", "")
            label_row = label_lookup.get(patient_id)
            if task_name in ("Task I", "Task II") and not self.is_patient_eligible_for_task(label_row, task_name):
                continue
            condition_text = self.format_patient_condition_text(label_row)
            display_text = f"{patient_id} - {condition_text}"
            self.patient_input.addItem(display_text, patient_id)

        if self.patient_input.count() == 0:
            self.patient_input.addItem(f"No eligible patients for {task_name}", "")

        if current_patient_id is not None:
            matched_index = self.patient_input.findData(current_patient_id)
            if matched_index >= 0:
                self.patient_input.setCurrentIndex(matched_index)
        elif self.patient_input.count() > 0:
            self.patient_input.setCurrentIndex(0)

        self.patient_input.blockSignals(False)

    def get_selected_patient_id(self):
        if isinstance(self.patient_input, QComboBox):
            patient_id = self.patient_input.currentData()
            if patient_id:
                return str(patient_id).strip()

            current_text = self.patient_input.currentText().strip()
            if current_text.startswith("No eligible patients"):
                return ""
            if " - " in current_text:
                return current_text.split(" - ", 1)[0].strip()
            return current_text

        return self.patient_input.text().strip()

    def get_selected_annotation_source(self):
        if not hasattr(self, 'annotation_selector'):
            return "ECG"
        if self.annotation_selector.currentIndex() == 1:
            return "AO"
        return "ECG"

    def get_dataset_class_summary(self):
        if self.dataset_class_summary_cache is not None:
            return self.dataset_class_summary_cache

        csv_paths = sorted(glob.glob(os.path.join(self.loader.data_dir, "Cleaned_*.csv")))
        available_patient_ids = {
            os.path.basename(path).replace("Cleaned_", "").replace(".csv", "")
            for path in csv_paths
        }

        labels_path = os.path.join(self.loader.data_dir, "ground_truth_labels.csv")
        df_labels = pd.read_csv(labels_path, sep=',')
        df_labels.columns = df_labels.columns.str.strip()
        df_labels = df_labels[df_labels['Patient ID'].isin(available_patient_ids)]

        label_rows = {
            row['Patient ID']: row.to_dict()
            for _, row in df_labels.iterrows()
        }

        summary = {
            'total_cleaned_files': len(available_patient_ids),
            'labeled_patients': len(label_rows),
            'missing_label_rows': int(len(available_patient_ids) - len(label_rows)),
            'tasks': {}
        }

        for task_name in ("Task I", "Task II", "Task III"):
            class_names = self.loader.get_task_class_names(task_name)
            class_counts = np.zeros(len(class_names), dtype=np.int64)
            excluded_by_definition = 0
            normal_count = 0
            included_patients = 0

            for patient_id in sorted(available_patient_ids):
                row = label_rows.get(patient_id)
                if row is None:
                    continue
                if task_name == "Task III":
                    multi_hot = self.loader.map_label_row_to_task_index(row, task_name)
                    class_counts += multi_hot.astype(np.int64)
                    included_patients += 1
                    if multi_hot.sum() == 0:
                        normal_count += 1
                else:
                    class_index = self.loader.map_label_row_to_task_index(row, task_name)
                    if class_index is None:
                        excluded_by_definition += 1
                    else:
                        class_counts[class_index] += 1
                        included_patients += 1

            summary['tasks'][task_name] = {
                'class_names': class_names,
                'class_counts': class_counts.tolist(),
                'included_patients': int(included_patients),
                'excluded_by_definition': int(excluded_by_definition),
                'normal_count': int(normal_count),
            }

        self.dataset_class_summary_cache = summary
        return summary

    def reset_pipeline_state(self):
        self.current_data = None
        self.segments = []
        self.preprocessed_segments = []
        self.current_segment_idx = 0
        self.scnn_models = {}
        self.lstm_models = {}
        self.sa_models = {}
        self.hvdnet_model = None
        self.training_worker = None
        self.train_loader = None
        self.class_weights = None
        self.current_training_task = self.task_selector.currentText()
        self.last_inference_result = None
        self.reset_training_diagnostics()
        self.clear_plots()
        self.clear_inference_plots()
        self.update_step_controls()
        self.update_navigation_controls()

    def reset_training_diagnostics(self):
        self.train_epochs = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_steps = []
        self.last_training_summary = None
        self.loaded_model_task = None
        self.loaded_model_classes = None
        self.loss_train_curve.setData([], [])
        self.loss_val_curve.setData([], [])
        self.acc_train_curve.setData([], [])
        self.acc_val_curve.setData([], [])
        self.cm_plot.clear()

    def update_step_controls(self):
        has_data = self.current_data is not None
        has_filtered = has_data and ('filtered_signals' in self.current_data)
        has_segments = len(self.segments) > 0
        has_preprocessed = len(self.preprocessed_segments) > 0
        has_cnn_outputs = has_data and ('cnn_outputs' in self.current_data)
        has_lstm_outputs = has_data and ('lstm_outputs' in self.current_data)
        has_sa_outputs = has_data and ('sa_outputs' in self.current_data)
        is_training = self.training_worker is not None and self.training_worker.isRunning()

        self.step2_btn.setEnabled(has_data)
        self.step3_btn.setEnabled(has_filtered)
        self.step4_btn.setEnabled(has_segments)
        self.step5_btn.setEnabled(has_preprocessed)
        self.step6_btn.setEnabled(has_cnn_outputs)
        self.step7_btn.setEnabled(has_lstm_outputs)
        self.step8_btn.setEnabled(has_sa_outputs)
        self.step9_btn.setEnabled(not is_training)
        self.pause_resume_btn.setEnabled(is_training)
        self.pause_resume_btn.setText("Resume Training" if self.is_training_paused else "Pause Training")
        self.infer_btn.setEnabled(self.hvdnet_model is not None and not is_training)
        self.cm_btn.setEnabled(self.hvdnet_model is not None and not is_training)
        self.class_attn_btn.setEnabled(self.hvdnet_model is not None and not is_training)
        self.save_model_btn.setEnabled(self.hvdnet_model is not None and not is_training)

    def step_toggle_training_pause(self):
        if self.training_worker is None or not self.training_worker.isRunning():
            self.log("[INFO] Training is not running.")
            self.is_training_paused = False
            self.update_step_controls()
            return

        if self.is_training_paused:
            self.training_worker.resume_training()
            self.is_training_paused = False
            self.log("[INFO] Training resumed.")
        else:
            self.training_worker.pause_training()
            self.is_training_paused = True
            self.log("[INFO] Training paused safely. Click again to resume.")

        self.update_step_controls()

    def step_load_data(self):
        patient_id = self.get_selected_patient_id()
        annotation_source = self.get_selected_annotation_source()
        self.console.clear()
        self.log(f"--- Step 1: Load Data for {patient_id} ---")

        self.reset_pipeline_state()
        self.current_patient_id = patient_id
        
        try:
            data = self.loader.load_patient_data(patient_id, annotation_source=annotation_source)

            self.log("[SUCCESS] Signals Loaded & Standardized:")
            self.log(f" > Target Sampling Rate: {data['fs']} Hz")
            self.log(f" > Total Samples per axis: {data['signal_length']}")

            self.log(f"\n[SUCCESS] {data.get('peak_label', 'Peaks')} Loaded:")
            self.log(f" > Found {len(data['r_peaks_indices'])} peaks.")
            self.log(f" > First 5 Peak Indices: {data['r_peaks_indices'][:5]}")
            self.log(f" > Source: {data.get('peak_source', annotation_source)}")

            self.log("\n[SUCCESS] Ground Truth Labels:")
            if isinstance(data['labels'], dict):
                for key, val in data['labels'].items():
                    self.log(f" > {key}: {val}")

                selected_task = self.task_selector.currentText()
                class_names = self.loader.get_task_class_names(selected_task)
                if selected_task == "Task III":
                    label_vector = data.get('label_vector')
                    if label_vector is not None:
                        positive_names = [
                            name for name, value in zip(class_names, label_vector)
                            if value > 0
                        ]
                        pretty_labels = ", ".join(positive_names) if positive_names else "Normal"
                        self.log(f" > Task Mapping ({selected_task}): {pretty_labels}")
                        self.log(f" > Multi-hot vector: {label_vector.astype(int).tolist()}")
                else:
                    mapped_idx = self.loader.map_label_row_to_task_index(data['labels'], selected_task)
                    if mapped_idx is not None:
                        self.log(f" > Task Mapping ({selected_task}): class {mapped_idx} ({class_names[mapped_idx]})")
                    else:
                        self.log(f" > Task Mapping ({selected_task}): excluded for this task definition")
            else:
                self.log(f" > {data['labels']}")

            preview_df = pd.DataFrame(data['signals'])
            self.log("\n[DATA PREVIEW] First 10 rows of raw/resampled signals:")
            self.log(preview_df.head(10).to_string(index=False))

            self.current_data = data

            self.plot_stage.setCurrentIndex(0)
            self.update_navigation_controls()
            self.update_step_controls()
            self.plot_current_view()

            self.log("\n[INTERACTIVE PLOT] Mouse wheel = zoom, left-drag = pan, right-drag = region zoom.")
            self.log("\nNext: click 'Step 2: Filter (1-30 Hz)'.")
            
        except Exception as e:
            self.reset_pipeline_state()
            self.log(f"\n[ERROR] {str(e)}")

    def step_apply_filter(self):
        if self.current_data is None:
            self.log("\n[INFO] Please run Step 1 first.")
            return

        self.log("\n--- Step 2: Apply Zero-Phase Butterworth Filter ---")
        try:
            self.current_data['filtered_signals'] = self.apply_zero_phase_butterworth(
                self.current_data['signals'],
                self.current_data['fs'],
                lowcut=1.0,
                highcut=30.0,
                order=6
            )
            self.plot_stage.setCurrentIndex(1)
            self.update_step_controls()
            self.plot_current_view()
            self.log("[SUCCESS] Zero-phase IIR Butterworth applied (order=6, band=1-30 Hz).")
            self.log("Next: click 'Step 3: Build Segments'.")
        except Exception as e:
            self.log(f"[ERROR] Filtering failed: {str(e)}")

    def step_build_segments(self):
        if self.current_data is None or 'filtered_signals' not in self.current_data:
            self.log("\n[INFO] Please run Step 2 first.")
            return

        self.log("\n--- Step 3: Build Peak-Based Segments ---")
        use_physio_filter = self.peak_filter_selector.currentIndex() == 1
        source_peaks = np.asarray(self.current_data['r_peaks_indices'], dtype=int)

        if use_physio_filter:
            # Build segments using 3-beat windows and filter by BPM (40-120 BPM).
            segments, discarded_segments = self.build_threebeat_segments(
                source_peaks,
                self.current_data['signal_length'],
                fs=self.current_data['fs'],
                min_bpm=40,
                max_bpm=120,
            )
            self.segments = segments
            self.current_data['segment_peak_indices'] = source_peaks  # keep original peaks intact
            self.current_data['segment_peak_discarded_count'] = int(discarded_segments)
        else:
            self.current_data['segment_peak_indices'] = source_peaks
            self.current_data['segment_peak_discarded_count'] = 0
            self.segments = self.build_rpeak_segments(
                source_peaks,
                self.current_data['signal_length']
            )
        self.preprocessed_segments = []
        self.current_segment_idx = 0
        self.view_mode.setCurrentIndex(1)
        self.update_navigation_controls()
        self.update_step_controls()
        self.plot_current_view()

        self.log("[SUCCESS] Segmentation complete.")
        self.log(" > Segment rule: P_i to P_{i+3}")
        if use_physio_filter:
            self.log(" > Physiological filter: ON (interval range 0.50s to 1.50s -> 40-120 BPM)")
            self.log(f" > Segments discarded as physiologically impossible: {self.current_data['segment_peak_discarded_count']}")
            self.log(f" > Peaks used for segmentation: {len(source_peaks)}")
        else:
            self.log(" > Physiological filter: OFF")
            self.log(" > Peaks discarded as physiologically impossible: 0")
        self.log(f" > Total segments: {len(self.segments)}")
        self.log("Next: click 'Step 4: Z-Score + Pad/Truncate 800'.")

    def zscore_normalize(self, values):
        values = np.asarray(values, dtype=float)
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val < 1e-12:
            return np.zeros_like(values)
        return (values - mean_val) / std_val

    def pad_or_truncate(self, values, target_len=800):
        values = np.asarray(values, dtype=float)
        if len(values) < target_len:
            return np.pad(values, (0, target_len - len(values)), mode='constant')
        if len(values) > target_len:
            return values[:target_len]
        return values

    def build_attention_sample_from_segment(self, filtered_signals, segment):
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']

        seg_x = self.pad_or_truncate(
            self.zscore_normalize(filtered_signals['AccX'][start_idx:end_idx]),
            target_len=800,
        )
        seg_y = self.pad_or_truncate(
            self.zscore_normalize(filtered_signals['AccY'][start_idx:end_idx]),
            target_len=800,
        )
        seg_z = self.pad_or_truncate(
            self.zscore_normalize(filtered_signals['AccZ'][start_idx:end_idx]),
            target_len=800,
        )

        x_tensor = torch.tensor(seg_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y_tensor = torch.tensor(seg_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        z_tensor = torch.tensor(seg_z, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return x_tensor, y_tensor, z_tensor

    def get_attention_sample_for_patient(self, patient_id, task_name):
        use_loaded_patient = self.current_patient_id == patient_id and self.current_data is not None
        data = self.current_data if use_loaded_patient else self.loader.load_patient_data(
            patient_id,
            annotation_source=self.get_selected_annotation_source(),
        )

        if 'filtered_signals' not in data:
            data['filtered_signals'] = self.apply_zero_phase_butterworth(
                data['signals'],
                data['fs'],
                lowcut=1.0,
                highcut=30.0,
                order=6,
            )

        if 'segments' not in data or not data['segments']:
            data['segments'] = self.build_rpeak_segments(
                data['r_peaks_indices'],
                data['signal_length'],
            )

        if not data['segments']:
            raise RuntimeError(f"No segments available for patient {patient_id}.")

        if use_loaded_patient and 'preprocessed_segments' in data and data['preprocessed_segments']:
            segment_idx = min(self.current_segment_idx, len(data['preprocessed_segments']) - 1)
            segment = data['preprocessed_segments'][segment_idx]
            x_tensor = torch.tensor(segment['AccX'], dtype=torch.float32).unsqueeze(0)
            y_tensor = torch.tensor(segment['AccY'], dtype=torch.float32).unsqueeze(0)
            z_tensor = torch.tensor(segment['AccZ'], dtype=torch.float32).unsqueeze(0)
        else:
            segment_idx = min(self.current_segment_idx, len(data['segments']) - 1) if use_loaded_patient else 0
            segment = data['segments'][segment_idx]
            x_tensor, y_tensor, z_tensor = self.build_attention_sample_from_segment(
                data['filtered_signals'],
                segment,
            )

        label_row = data.get('labels')
        if task_name == "Task III":
            true_label = data.get('label_vector')
        else:
            true_label = self.loader.map_label_row_to_task_index(label_row, task_name)

        return {
            'patient_id': patient_id,
            'segment_idx': segment_idx,
            'segment': segment,
            'x_tensor': x_tensor,
            'y_tensor': y_tensor,
            'z_tensor': z_tensor,
            'true_label': true_label,
            'label_row': label_row,
            'class_names': self.loader.get_task_class_names(task_name),
        }

    def get_all_task3_segments_for_patient(self, patient_id):
        use_loaded_patient = self.current_patient_id == patient_id and self.current_data is not None
        data = self.current_data if use_loaded_patient else self.loader.load_patient_data(
            patient_id,
            annotation_source=self.get_selected_annotation_source(),
        )

        if 'filtered_signals' not in data:
            data['filtered_signals'] = self.apply_zero_phase_butterworth(
                data['signals'],
                data['fs'],
                lowcut=1.0,
                highcut=30.0,
                order=6,
            )

        if 'segments' not in data or not data['segments']:
            data['segments'] = self.build_rpeak_segments(
                data['r_peaks_indices'],
                data['signal_length'],
            )

        if not data['segments']:
            raise RuntimeError(f"No segments available for patient {patient_id}.")

        x_samples = []
        y_samples = []
        z_samples = []

        for segment in data['segments']:
            x_tensor, y_tensor, z_tensor = self.build_attention_sample_from_segment(
                data['filtered_signals'],
                segment,
            )
            x_samples.append(x_tensor)
            y_samples.append(y_tensor)
            z_samples.append(z_tensor)

        label_vector = data.get('label_vector')
        if label_vector is None:
            raise RuntimeError(f"Missing Task III label vector for patient {patient_id}.")

        return {
            'patient_id': patient_id,
            'x_batch': torch.cat(x_samples, dim=0),
            'y_batch': torch.cat(y_samples, dim=0),
            'z_batch': torch.cat(z_samples, dim=0),
            'label_vector': np.asarray(label_vector, dtype=float),
            'num_segments': len(x_samples),
            'class_names': self.loader.get_task_class_names("Task III"),
        }

    def run_task3_patient_level_aggregation(self, patient_id, model, device):
        patient_data = self.get_all_task3_segments_for_patient(patient_id)

        model.eval()
        with torch.no_grad():
            logits, _ = model(
                move_tensor_to_device(patient_data['x_batch'], device),
                move_tensor_to_device(patient_data['y_batch'], device),
                move_tensor_to_device(patient_data['z_batch'], device),
            )
            segment_probabilities = torch.sigmoid(logits).cpu().numpy()

        mean_probabilities = np.mean(segment_probabilities, axis=0)
        patient_prediction = (mean_probabilities > 0.5).astype(float)
        actual_truth = patient_data['label_vector']

        return {
            'patient_id': patient_data['patient_id'],
            'num_segments': patient_data['num_segments'],
            'segment_probabilities': segment_probabilities,
            'mean_probabilities': mean_probabilities,
            'patient_prediction': patient_prediction,
            'actual_truth': actual_truth,
            'is_exact_match': np.array_equal(patient_prediction, actual_truth),
            'class_names': patient_data['class_names'],
        }

    def step_preprocess_segments(self):
        if not self.segments:
            self.log("\n[INFO] Please run Step 3 first.")
            return

        self.log("\n--- Step 4: Z-Score + Zero-Pad/Truncate + Reshape ---")
        source = self.current_data['filtered_signals']
        target_len = 800
        preprocessed = []

        for seg in self.segments:
            start_idx = seg['start_idx']
            end_idx = seg['end_idx']

            seg_item = {
                'segment_id': seg['segment_id'],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_peak_number': seg['start_peak_number'],
                'end_peak_number': seg['end_peak_number']
            }

            for axis in ('AccX', 'AccY', 'AccZ'):
                raw_seg = source[axis][start_idx:end_idx]
                z_seg = self.zscore_normalize(raw_seg)
                fixed_seg = self.pad_or_truncate(z_seg, target_len=target_len)
                seg_item[axis] = fixed_seg.reshape(1, target_len)

            ecg_seg = source['ECG'][start_idx:end_idx]
            seg_item['ECG_fixed'] = self.pad_or_truncate(ecg_seg, target_len=target_len)
            preprocessed.append(seg_item)

        self.preprocessed_segments = preprocessed
        self.current_data['preprocessed_segments'] = preprocessed

        accx_batch = np.stack([s['AccX'] for s in preprocessed], axis=0)
        accy_batch = np.stack([s['AccY'] for s in preprocessed], axis=0)
        accz_batch = np.stack([s['AccZ'] for s in preprocessed], axis=0)
        scg_3axis_batch = np.concatenate([accx_batch, accy_batch, accz_batch], axis=1)

        self.current_data['pytorch_inputs'] = {
            'AccX': accx_batch,
            'AccY': accy_batch,
            'AccZ': accz_batch,
            'SCG_3AXIS': scg_3axis_batch
        }

        self.plot_stage.setCurrentIndex(2)
        self.view_mode.setCurrentIndex(1)
        self.current_segment_idx = 0
        self.update_navigation_controls()
        self.plot_current_view()

        self.log("[SUCCESS] Z-score normalization complete for all SCG segment axes.")
        self.log("[SUCCESS] Each segment padded/truncated to exactly 800 samples.")
        self.log("[SUCCESS] PyTorch-ready shapes prepared:")
        self.log(f" > AccX batch: {accx_batch.shape} (batch, 1, 800)")
        self.log(f" > AccY batch: {accy_batch.shape} (batch, 1, 800)")
        self.log(f" > AccZ batch: {accz_batch.shape} (batch, 1, 800)")
        self.log(f" > Combined SCG batch: {scg_3axis_batch.shape} (batch, 3, 800)")
        self.update_step_controls()
        self.log("Next: click 'Step 5: Build sCNN + Forward Check'.")

    def step_build_cnn(self):
        if self.current_data is None or 'pytorch_inputs' not in self.current_data:
            self.log("\n[INFO] Please run Step 4 first.")
            return

        self.log("\n--- Step 5: Build Axis-Specific sCNN Branches (Late Fusion) ---")

        x_tensor = torch.tensor(self.current_data['pytorch_inputs']['AccX'], dtype=torch.float32)
        y_tensor = torch.tensor(self.current_data['pytorch_inputs']['AccY'], dtype=torch.float32)
        z_tensor = torch.tensor(self.current_data['pytorch_inputs']['AccZ'], dtype=torch.float32)

        self.scnn_models = {
            'x': SCNN_Module(in_channels=1, base_filters=64, kernel_size=7),
            'y': SCNN_Module(in_channels=1, base_filters=64, kernel_size=7),
            'z': SCNN_Module(in_channels=1, base_filters=64, kernel_size=7)
        }

        for model in self.scnn_models.values():
            model.eval()

        with torch.no_grad():
            x_features = self.scnn_models['x'](x_tensor)
            y_features = self.scnn_models['y'](y_tensor)
            z_features = self.scnn_models['z'](z_tensor)

        self.current_data['cnn_outputs'] = {
            'x_features': x_features,
            'y_features': y_features,
            'z_features': z_features
        }

        self.log("[SUCCESS] Axis-specific sCNN branches initialized (late fusion compatible).")
        self.log("[CONFIG] Per-branch channel flow: 64 -> 32 -> 16")
        self.log(f" > X input shape: {tuple(x_tensor.shape)} -> output: {tuple(x_features.shape)}")
        self.log(f" > Y input shape: {tuple(y_tensor.shape)} -> output: {tuple(y_features.shape)}")
        self.log(f" > Z input shape: {tuple(z_tensor.shape)} -> output: {tuple(z_features.shape)}")
        self.log("[EXPECTED] With input length 800 and 3x MaxPool(2), output length is 100.")
        self.log(" > Per-axis output target: (batch, 16, 100)")
        self.log(" > These three outputs are ready for axis-specific LSTM modules.")
        self.update_step_controls()
        self.log("Next: click 'Step 6: Build LSTM + Forward Check'.")

    def step_build_lstm(self):
        if self.current_data is None or 'cnn_outputs' not in self.current_data:
            self.log("\n[INFO] Please run Step 5 first.")
            return

        self.log("\n--- Step 6 (Verification): Axis-Specific LSTM Forward Check ---")

        x_features = self.current_data['cnn_outputs']['x_features']
        y_features = self.current_data['cnn_outputs']['y_features']
        z_features = self.current_data['cnn_outputs']['z_features']

        self.lstm_models = {
            'x': LSTM_Module(input_features=16, hidden_size=64),
            'y': LSTM_Module(input_features=16, hidden_size=64),
            'z': LSTM_Module(input_features=16, hidden_size=64)
        }

        for model in self.lstm_models.values():
            model.eval()

        with torch.no_grad():
            x_lstm_out = self.lstm_models['x'](x_features)
            y_lstm_out = self.lstm_models['y'](y_features)
            z_lstm_out = self.lstm_models['z'](z_features)

        self.current_data['lstm_outputs'] = {
            'x_lstm_out': x_lstm_out,
            'y_lstm_out': y_lstm_out,
            'z_lstm_out': z_lstm_out
        }

        self.log("[SUCCESS] Three axis-specific LSTM modules initialized.")
        self.log("[CONFIG] input_features=16, hidden_size=64, num_layers=1, batch_first=True")
        self.log("[CHECKPOINT] X-axis verification:")
        self.log(f" > X sCNN input to LSTM: {tuple(x_features.shape)} (batch, 16, 100)")
        self.log(f" > X LSTM output shape: {tuple(x_lstm_out.shape)} (batch, 100, 64)")
        self.log(f" > Y LSTM output shape: {tuple(y_lstm_out.shape)}")
        self.log(f" > Z LSTM output shape: {tuple(z_lstm_out.shape)}")
        self.log(" > Outputs are ready for axis-specific self-attention modules.")
        self.update_step_controls()
        self.log("Next: click 'Step 7: Build SA + Forward Check'.")

    def step_build_sa(self):
        if self.current_data is None or 'lstm_outputs' not in self.current_data:
            self.log("\n[INFO] Please run Step 6 first.")
            return

        self.log("\n--- Step 7: Build Axis-Specific Self-Attention Modules ---")

        x_lstm_out = self.current_data['lstm_outputs']['x_lstm_out']
        y_lstm_out = self.current_data['lstm_outputs']['y_lstm_out']
        z_lstm_out = self.current_data['lstm_outputs']['z_lstm_out']

        self.sa_models = {
            'x': SA_Module(hidden_size=64),
            'y': SA_Module(hidden_size=64),
            'z': SA_Module(hidden_size=64)
        }

        for model in self.sa_models.values():
            model.eval()

        with torch.no_grad():
            x_context, x_weights = self.sa_models['x'](x_lstm_out)
            y_context, y_weights = self.sa_models['y'](y_lstm_out)
            z_context, z_weights = self.sa_models['z'](z_lstm_out)

        self.current_data['sa_outputs'] = {
            'x_context': x_context,
            'y_context': y_context,
            'z_context': z_context,
            'x_weights': x_weights,
            'y_weights': y_weights,
            'z_weights': z_weights
        }

        self.log("[SUCCESS] Axis-specific SA modules initialized.")
        self.log("[CHECKPOINT] Attention verification:")
        self.log(f" > X LSTM input to SA: {tuple(x_lstm_out.shape)} (batch, 100, 64)")
        self.log(f" > X context vector shape: {tuple(x_context.shape)} (batch, 64)")
        self.log(f" > Y context vector shape: {tuple(y_context.shape)} (batch, 64)")
        self.log(f" > Z context vector shape: {tuple(z_context.shape)} (batch, 64)")
        self.log(" > Raw attention weights are stored for later heatmap generation.")
        self.update_step_controls()
        self.log("Next: click 'Step 8: Build HVDNet + Final Check'.")

    def step_build_hvdnet(self):
        self.log("\n--- Step 8: Full HVDNet Forward Check ---")

        task_name = self.task_selector.currentText()
        class_names = self.loader.get_task_class_names(task_name)

        self.hvdnet_model = HVDNet(num_classes=5, d=64)
        self.hvdnet_model.eval()

        dummy_x = torch.randn(520, 1, 800)
        dummy_y = torch.randn(520, 1, 800)
        dummy_z = torch.randn(520, 1, 800)

        with torch.no_grad():
            logits, attention_weights = self.hvdnet_model(dummy_x, dummy_y, dummy_z)

        attn_x, attn_y, attn_z = attention_weights

        self.current_data['hvdnet_outputs'] = {
            'logits': logits,
            'attention_weights': attention_weights
        }

        self.log("[SUCCESS] HVDNet initialized with late-fusion axis-specific branches.")
        self.log("[CHECKPOINT] Full network forward pass:")
        self.log(f" > dummy_x shape: {tuple(dummy_x.shape)}")
        self.log(f" > dummy_y shape: {tuple(dummy_y.shape)}")
        self.log(f" > dummy_z shape: {tuple(dummy_z.shape)}")
        self.log(f" > logits shape: {tuple(logits.shape)} (batch, 5)")
        self.log(f" > attention_x shape: {tuple(attn_x.shape)}")
        self.log(f" > attention_y shape: {tuple(attn_y.shape)}")
        self.log(f" > attention_z shape: {tuple(attn_z.shape)}")
        self.log(f" > Active task: {task_name} | Classes: {class_names}")

    def build_training_dataset(self, task_name):
        class_names = self.loader.get_task_class_names(task_name)

        csv_paths = sorted(glob.glob(os.path.join(self.loader.data_dir, "Cleaned_*.csv")))
        if not csv_paths:
            raise RuntimeError(f"No training CSV files found in {self.loader.data_dir}")

        x_samples = []
        y_samples = []
        z_samples = []
        labels = []
        skipped_by_task = 0
        skipped_no_segments = 0
        skipped_errors = 0
        class_counts = np.zeros(len(class_names), dtype=np.int64)

        for csv_path in csv_paths:
            patient_id = os.path.basename(csv_path).replace("Cleaned_", "").replace(".csv", "")
            try:
                patient_data = self.loader.load_patient_data(
                    patient_id,
                    annotation_source=self.get_selected_annotation_source(),
                )
                if not isinstance(patient_data.get('labels'), dict):
                    skipped_by_task += 1
                    continue

                if task_name == "Task III":
                    label_value = self.loader.map_label_row_to_task_index(patient_data['labels'], task_name)
                else:
                    class_index = self.loader.map_label_row_to_task_index(patient_data['labels'], task_name)

                    if class_index is None:
                        skipped_by_task += 1
                        continue

                patient_data['filtered_signals'] = self.apply_zero_phase_butterworth(
                    patient_data['signals'],
                    patient_data['fs'],
                    lowcut=1.0,
                    highcut=30.0,
                    order=6,
                )

                segments = self.build_rpeak_segments(
                    patient_data['r_peaks_indices'],
                    patient_data['signal_length'],
                )
                if not segments:
                    skipped_no_segments += 1
                    continue

                for seg in segments:
                    start_idx = seg['start_idx']
                    end_idx = seg['end_idx']

                    seg_x = self.pad_or_truncate(
                        self.zscore_normalize(patient_data['filtered_signals']['AccX'][start_idx:end_idx]),
                        target_len=800,
                    )
                    seg_y = self.pad_or_truncate(
                        self.zscore_normalize(patient_data['filtered_signals']['AccY'][start_idx:end_idx]),
                        target_len=800,
                    )
                    seg_z = self.pad_or_truncate(
                        self.zscore_normalize(patient_data['filtered_signals']['AccZ'][start_idx:end_idx]),
                        target_len=800,
                    )

                    x_samples.append(torch.tensor(seg_x, dtype=torch.float32).unsqueeze(0))
                    y_samples.append(torch.tensor(seg_y, dtype=torch.float32).unsqueeze(0))
                    z_samples.append(torch.tensor(seg_z, dtype=torch.float32).unsqueeze(0))
                    if task_name == "Task III":
                        labels.append(torch.tensor(label_value, dtype=torch.float32))
                        class_counts += label_value.astype(np.int64)
                    else:
                        labels.append(class_index)
                        class_counts[class_index] += 1

            except Exception:
                skipped_errors += 1

        if not x_samples:
            raise RuntimeError("No training samples were created. Check your label mapping and segment extraction.")

        x_tensor = torch.stack(x_samples, dim=0)
        y_tensor = torch.stack(y_samples, dim=0)
        z_tensor = torch.stack(z_samples, dim=0)
        if task_name == "Task III":
            label_tensor = torch.stack(labels, dim=0).to(dtype=torch.float32)
        else:
            label_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            'x_tensor': x_tensor,
            'y_tensor': y_tensor,
            'z_tensor': z_tensor,
            'label_tensor': label_tensor,
            'num_samples': len(label_tensor),
            'class_counts': class_counts,
            'skipped_by_task': skipped_by_task,
            'skipped_no_segments': skipped_no_segments,
            'skipped_errors': skipped_errors,
            'class_names': class_names,
        }

    def step_train_model(self):
        self.log("\n--- Step 9: Training Loop ---")
        try:
            self.reset_training_diagnostics()

            summary = self.get_dataset_class_summary()
            self.log("[DATA RECAP] All cleaned files before training:")
            self.log(
                f" > Total cleaned files: {summary['total_cleaned_files']} | "
                f"Labeled: {summary['labeled_patients']} | "
                f"Missing label rows: {summary['missing_label_rows']}"
            )
            for task_name in ("Task I", "Task II", "Task III"):
                task_info = summary['tasks'][task_name]
                class_parts = [
                    f"{name}={count}"
                    for name, count in zip(task_info['class_names'], task_info['class_counts'])
                ]
                extra_parts = []
                if task_name == "Task III":
                    extra_parts.append(f"Normal={task_info['normal_count']}")
                self.log(
                    f" > {task_name}: "
                    + ", ".join(class_parts)
                    + (" | " + ", ".join(extra_parts) if extra_parts else "")
                    + (
                        f" | Included={task_info['included_patients']}"
                        f" | Excluded by definition={task_info['excluded_by_definition']}"
                    )
                )

            task_name = self.task_selector.currentText()
            self.current_training_task = task_name
            dataset_info = self.build_training_dataset(task_name)

            is_small_training = self.small_training_mode.currentIndex() == 1
            num_epochs = 50 if is_small_training else 100
            n_splits = 3 if is_small_training else 5
            self.current_num_epochs = num_epochs
            self.current_n_splits = n_splits

            self.log("[SUCCESS] Training dataset prepared from all available patient files.")
            self.log(f" > Task: {task_name}")
            self.log(f" > Task classes: {dataset_info['class_names']}")
            self.log(f" > Total samples: {dataset_info['num_samples']}")
            self.log(f" > Class counts: {dataset_info['class_counts'].tolist()}")
            self.log(f" > Skipped rows excluded by task definition: {dataset_info['skipped_by_task']}")
            self.log(f" > Skipped patients with no segments: {dataset_info['skipped_no_segments']}")
            self.log(f" > Skipped errors: {dataset_info['skipped_errors']}")
            self.training_worker = TrainingWorker(
                x_tensor=dataset_info['x_tensor'],
                y_tensor=dataset_info['y_tensor'],
                z_tensor=dataset_info['z_tensor'],
                label_tensor=dataset_info['label_tensor'],
                num_classes=5,
                d=64,
                num_epochs=num_epochs,
                batch_size=64,
                learning_rate=0.001,
                weight_decay=0.004,
                test_size=0.2,
                n_splits=n_splits,
                random_state=42,
                multi_label=(task_name == "Task III"),
            )
            self.training_worker.log_update.connect(self.log)
            self.training_worker.epoch_update.connect(self.on_training_epoch_update)
            self.training_worker.test_update.connect(self.on_test_metrics_update)
            self.training_worker.error_update.connect(self.on_training_error)
            self.training_worker.finished_update.connect(self.on_training_finished)
            self.training_worker.start()
            self.is_training_paused = False

            self.log("[INFO] Training started in a background thread to keep the UI responsive.")
            if task_name == "Task III":
                self.log(f"[INFO] Split strategy: 80% train pool / 20% held-out test, then {n_splits}-fold KFold on train pool.")
            else:
                self.log(f"[INFO] Split strategy: 80% train pool / 20% held-out test, then {n_splits}-fold stratified CV on train pool.")
            self.update_step_controls()
        except Exception as e:
            self.log(f"[ERROR] Training setup failed: {str(e)}")

    def format_seconds(self, seconds):
        seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def on_training_epoch_update(self, fold, epoch, train_loss, train_acc, val_loss, val_acc, epoch_seconds, eta_seconds):
        step = len(self.epoch_steps) + 1
        self.epoch_steps.append(step)
        self.train_epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        self.loss_train_curve.setData(self.epoch_steps, self.train_losses)
        self.loss_val_curve.setData(self.epoch_steps, self.val_losses)
        self.acc_train_curve.setData(self.epoch_steps, self.train_accuracies)
        self.acc_val_curve.setData(self.epoch_steps, self.val_accuracies)

        epoch_time_text = self.format_seconds(epoch_seconds)
        eta_text = self.format_seconds(eta_seconds)
        self.log(
            f"Fold [{fold}/{self.current_n_splits}] Epoch [{epoch}/{self.current_num_epochs}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"Epoch Time: {epoch_time_text} | ETA: {eta_text}"
        )

    def on_test_metrics_update(self, test_loss, test_acc):
        self.log(f"[TEST] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

    def on_training_finished(self, summary):
        self.is_training_paused = False
        self.last_training_summary = summary
        self.hvdnet_model = HVDNet(num_classes=5, d=64)
        self.hvdnet_model.load_state_dict(summary['best_state_dict'])
        self.loaded_model_task = self.current_training_task
        self.loaded_model_classes = self.loader.get_task_class_names(self.current_training_task)
        self.task_selector.setCurrentText(self.current_training_task)
        self.refresh_class_attention_class_selector()

        self.log("[SUCCESS] Training complete.")
        self.log(f" > Best fold by validation loss: {summary['best_fold']}")
        self.log(f" > Best validation loss: {summary['best_val_loss']:.4f}")
        self.log(f" > Held-out test size: {summary['test_size']}")
        self.log(f" > Held-out test loss: {summary['test_loss']:.4f}")
        self.log(f" > Held-out test accuracy: {summary['test_acc']:.2f}%")
        self.update_step_controls()

    def on_training_error(self, message):
        self.is_training_paused = False
        self.log(f"[ERROR] Training failed: {message}")
        self.update_step_controls()

    def step_save_model(self):
        if self.hvdnet_model is None:
            self.log("[INFO] No model available to save. Train first (Step 9).")
            return
        if self.training_worker is not None and self.training_worker.isRunning():
            self.log("[INFO] Please wait for training to finish before saving.")
            return

        task_name = self.current_training_task
        class_names = self.loader.get_task_class_names(task_name)
        default_name = f"hvdnet_{task_name.replace(' ', '_').lower()}.pt"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trained Model",
            default_name,
            "PyTorch Model (*.pt *.pth)",
        )

        if not save_path:
            self.log("[INFO] Save canceled.")
            return

        state_dict_cpu = {k: v.detach().cpu() for k, v in self.hvdnet_model.state_dict().items()}
        payload = {
            'model_state_dict': state_dict_cpu,
            'num_classes': 5,
            'd': 64,
            'task_name': task_name,
            'class_names': class_names,
            'label_columns': self.loader.label_columns,
        }
        torch.save(payload, save_path)
        self.log(f"[SUCCESS] Model saved to: {save_path}")

    def step_load_saved_model(self):
        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Trained Model",
            "",
            "PyTorch Model (*.pt *.pth)",
        )
        if not load_path:
            self.log("[INFO] Load canceled.")
            return

        payload = torch.load(load_path, map_location='cpu')
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            state_dict = payload['model_state_dict']
            task_name = payload.get('task_name', self.task_selector.currentText())
            class_names = payload.get('class_names', self.loader.get_task_class_names(task_name))
        else:
            state_dict = payload
            task_name = self.task_selector.currentText()
            class_names = self.loader.get_task_class_names(task_name)

        self.hvdnet_model = HVDNet(num_classes=5, d=64)
        self.hvdnet_model.load_state_dict(state_dict)
        self.hvdnet_model.eval()

        self.loaded_model_task = task_name
        self.loaded_model_classes = class_names
        if task_name in ["Task I", "Task II", "Task III"]:
            self.task_selector.setCurrentText(task_name)
        self.refresh_class_attention_class_selector()

        self.log(f"[SUCCESS] Loaded model from: {load_path}")
        self.log(f" > Task: {task_name}")
        self.log(f" > Classes: {class_names}")
        self.update_step_controls()

    def get_task_test_split_tensors(self, task_name):
        dataset_info = self.build_training_dataset(task_name)
        x_tensor = dataset_info['x_tensor']
        y_tensor = dataset_info['y_tensor']
        z_tensor = dataset_info['z_tensor']
        label_tensor = dataset_info['label_tensor']

        labels_np = label_tensor.cpu().numpy()
        indices = np.arange(len(labels_np))
        if task_name == "Task III":
            _, test_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )
        else:
            _, test_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=labels_np,
            )

        return x_tensor[test_idx], y_tensor[test_idx], z_tensor[test_idx], label_tensor[test_idx], dataset_info['class_names']

    def plot_attention_overlay(self, signal_values, attention_values, plot_item, axis_name, title_prefix):
        plot_item.clear()

        signal_values = np.asarray(signal_values, dtype=float)
        attention_values = np.asarray(attention_values, dtype=float)
        if len(signal_values) == 0 or len(attention_values) == 0:
            return

        scale = len(signal_values) / max(len(attention_values), 1)
        attn_upsampled = scipy.ndimage.zoom(attention_values, scale, order=1)
        if len(attn_upsampled) < len(signal_values):
            attn_upsampled = np.pad(attn_upsampled, (0, len(signal_values) - len(attn_upsampled)), mode='edge')
        attn_upsampled = attn_upsampled[:len(signal_values)]

        y_min = float(np.min(signal_values) - 0.5)
        y_max = float(np.max(signal_values) + 0.5)
        x = np.arange(len(signal_values), dtype=float)

        # PyQtGraph ImageItem expects first axis as x-width; shape (N, 1)
        # makes attention vary along time (vertical stripes) instead of amplitude.
        heatmap_data = np.expand_dims(attn_upsampled, axis=1)
        image_item = pg.ImageItem(heatmap_data)
        image_item.setRect(QRectF(0, y_min, len(signal_values), max(y_max - y_min, 1e-6)))
        image_item.setOpacity(0.45)
        cmap = pg.colormap.get('CET-L4')
        image_item.setLookupTable(cmap.getLookupTable())
        image_item.setZValue(0)
        plot_item.addItem(image_item)

        curve = pg.PlotCurveItem(x, signal_values, pen=pg.mkPen('#d62728', width=2))
        curve.setZValue(1)
        plot_item.addItem(curve)
        plot_item.setTitle(f"{title_prefix} - {axis_name} Axis Attention")
        plot_item.setLabel('left', f'{axis_name} Amplitude')
        plot_item.setLabel('bottom', 'Samples')
        plot_item.setXRange(0, len(signal_values), padding=0)
        plot_item.setYRange(y_min, y_max, padding=0)
        plot_item.showGrid(x=True, y=True, alpha=0.25)

    def resize_attention_to_length(self, attention_values, target_len):
        attention_values = np.asarray(attention_values, dtype=float).reshape(-1)
        if target_len <= 0:
            return np.asarray([], dtype=float)
        if len(attention_values) == 0:
            return np.zeros(target_len, dtype=float)

        scale = target_len / max(len(attention_values), 1)
        resized = scipy.ndimage.zoom(attention_values, scale, order=1)
        if len(resized) < target_len:
            resized = np.pad(resized, (0, target_len - len(resized)), mode='edge')
        return resized[:target_len]

    def build_patient_inference_result(self, patient_id, task_name, model, device):
        use_loaded_patient = self.current_patient_id == patient_id and self.current_data is not None
        data = self.current_data if use_loaded_patient else self.loader.load_patient_data(
            patient_id,
            annotation_source=self.get_selected_annotation_source(),
        )

        if 'filtered_signals' not in data:
            data['filtered_signals'] = self.apply_zero_phase_butterworth(
                data['signals'],
                data['fs'],
                lowcut=1.0,
                highcut=30.0,
                order=6,
            )

        if 'segments' not in data or not data['segments']:
            data['segments'] = self.build_rpeak_segments(
                data['r_peaks_indices'],
                data['signal_length'],
            )

        if not data['segments']:
            raise RuntimeError(f"No segments available for patient {patient_id}.")

        if task_name == "Task III" and data.get('label_vector') is None:
            raise RuntimeError(f"Missing Task III label vector for patient {patient_id}.")

        segment_results = []
        signal_length = int(data['signal_length'])
        full_attention_sum = {
            'X': np.zeros(signal_length, dtype=float),
            'Y': np.zeros(signal_length, dtype=float),
            'Z': np.zeros(signal_length, dtype=float),
        }
        full_attention_count = {
            'X': np.zeros(signal_length, dtype=float),
            'Y': np.zeros(signal_length, dtype=float),
            'Z': np.zeros(signal_length, dtype=float),
        }

        model.eval()
        with torch.no_grad():
            for segment_idx, segment in enumerate(data['segments']):
                x_tensor, y_tensor, z_tensor = self.build_attention_sample_from_segment(
                    data['filtered_signals'],
                    segment,
                )
                logits, (attn_x, attn_y, attn_z) = model(
                    move_tensor_to_device(x_tensor, device),
                    move_tensor_to_device(y_tensor, device),
                    move_tensor_to_device(z_tensor, device),
                )

                segment_probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                if task_name == "Task III":
                    segment_prediction = (segment_probabilities > 0.5).astype(float)
                    true_label = np.asarray(data['label_vector'], dtype=float)
                else:
                    segment_prediction = int(torch.argmax(logits, dim=1).item())
                    true_label = self.loader.map_label_row_to_task_index(data['labels'], task_name)

                attn_arrays = {
                    'X': attn_x.squeeze().detach().cpu().numpy(),
                    'Y': attn_y.squeeze().detach().cpu().numpy(),
                    'Z': attn_z.squeeze().detach().cpu().numpy(),
                }

                segment_length = int(segment['end_idx'] - segment['start_idx'])
                for axis_name, attention_values in attn_arrays.items():
                    resized_attention = self.resize_attention_to_length(attention_values, segment_length)
                    start_idx = int(segment['start_idx'])
                    end_idx = int(segment['end_idx'])
                    full_attention_sum[axis_name][start_idx:end_idx] += resized_attention
                    full_attention_count[axis_name][start_idx:end_idx] += 1.0

                segment_results.append({
                    'segment_idx': segment_idx,
                    'segment': segment,
                    'start_idx': int(segment['start_idx']),
                    'end_idx': int(segment['end_idx']),
                    'segment_length': segment_length,
                    'x_tensor': x_tensor,
                    'y_tensor': y_tensor,
                    'z_tensor': z_tensor,
                    'signal_x': data['filtered_signals']['AccX'][segment['start_idx']:segment['end_idx']],
                    'signal_y': data['filtered_signals']['AccY'][segment['start_idx']:segment['end_idx']],
                    'signal_z': data['filtered_signals']['AccZ'][segment['start_idx']:segment['end_idx']],
                    'signal_ecg': data['filtered_signals']['ECG'][segment['start_idx']:segment['end_idx']],
                    'attention_x': attn_arrays['X'],
                    'attention_y': attn_arrays['Y'],
                    'attention_z': attn_arrays['Z'],
                    'probabilities': segment_probabilities,
                    'prediction': segment_prediction,
                    'true_label': true_label,
                })

        full_attention = {}
        for axis_name in ('X', 'Y', 'Z'):
            full_attention[axis_name] = np.divide(
                full_attention_sum[axis_name],
                full_attention_count[axis_name],
                out=np.zeros_like(full_attention_sum[axis_name]),
                where=full_attention_count[axis_name] > 0,
            )

        segment_probability_matrix = np.stack([item['probabilities'] for item in segment_results], axis=0)
        mean_probabilities = np.mean(segment_probability_matrix, axis=0)

        if task_name == "Task III":
            final_prediction = (mean_probabilities > 0.5).astype(float)
            actual_truth = np.asarray(data['label_vector'], dtype=float)
        else:
            vote_counts = np.bincount(
                np.asarray([int(item['prediction']) for item in segment_results], dtype=int),
                minlength=len(self.loader.get_task_class_names(task_name)),
            )
            final_prediction = int(np.argmax(vote_counts))
            actual_truth = self.loader.map_label_row_to_task_index(data['labels'], task_name)

        return {
            'patient_id': patient_id,
            'task_name': task_name,
            'data': data,
            'segment_results': segment_results,
            'full_attention': full_attention,
            'mean_probabilities': mean_probabilities,
            'final_prediction': final_prediction,
            'actual_truth': actual_truth,
            'is_exact_match': np.array_equal(final_prediction, actual_truth) if task_name == "Task III" else int(final_prediction) == int(actual_truth),
            'class_names': self.loader.get_task_class_names(task_name),
            'num_segments': len(segment_results),
        }

    def plot_inference_full_signal(self, patient_id, data, title_prefix, plot_targets=None, attention_maps=None):
        accx_plot, accy_plot, accz_plot, ecg_plot = plot_targets or (
            self.accx_plot,
            self.accy_plot,
            self.accz_plot,
            self.ecg_plot,
        )

        fs = data['fs']
        stage_index = self.plot_stage.currentIndex()
        stage_name = self.plot_stage.currentText()

        if stage_index == 0:
            signals = data['signals']
        else:
            signals = data.get('filtered_signals', data['signals'])

        accx = signals['AccX']
        accy = signals['AccY']
        accz = signals['AccZ']
        ecg = signals['ECG']
        peak_indices = np.asarray(data.get('r_peaks_indices', []), dtype=int)
        peak_label = data.get('peak_label', 'R-peaks')
        peak_plot_axis = data.get('peak_plot_axis', 'ECG')

        n = data['signal_length']
        t = np.arange(n) / fs

        accx_plot.clear()
        accy_plot.clear()
        accz_plot.clear()
        ecg_plot.clear()

        title = f"Patient {patient_id} - {stage_name} | {title_prefix}"
        if attention_maps is not None:
            self.plot_attention_overlay(accx, attention_maps['X'], accx_plot, 'X', title)
            self.plot_attention_overlay(accy, attention_maps['Y'], accy_plot, 'Y', title)
            self.plot_attention_overlay(accz, attention_maps['Z'], accz_plot, 'Z', title)
        else:
            accx_plot.setTitle(title)
            accx_plot.plot(t, accx, pen=pg.mkPen('#1f77b4', width=1))
            accy_plot.plot(t, accy, pen=pg.mkPen('#ff7f0e', width=1))
            accz_plot.plot(t, accz, pen=pg.mkPen('#2ca02c', width=1))
        ecg_plot.plot(t, ecg, pen=pg.mkPen('#d62728', width=1))

        if len(peak_indices):
            peak_times = peak_indices / fs
            if peak_plot_axis == 'AccZ':
                peak_plot_target = accz_plot
                peak_signal_values = accz
            else:
                peak_plot_target = ecg_plot
                peak_signal_values = ecg
            peak_vals = peak_signal_values[peak_indices]
            for peak_time in peak_times:
                peak_plot_target.addLine(x=float(peak_time), pen=pg.mkPen((255, 255, 0, 90), width=1))
            r_peak_scatter = pg.ScatterPlotItem(
                x=peak_times,
                y=peak_vals,
                pen=pg.mkPen((0, 0, 0), width=1),
                brush=pg.mkBrush(255, 255, 0),
                size=10,
                name=peak_label
            )
            peak_plot_target.addItem(r_peak_scatter)

        if len(t):
            initial_window_sec = min(10, t[-1])
            if initial_window_sec > 0:
                accx_plot.setXRange(0, initial_window_sec, padding=0.01)

    def step_run_inference_attention(self):
        if self.hvdnet_model is None:
            self.log("[INFO] Load or train a model first.")
            return

        task_name = self.loaded_model_task or self.task_selector.currentText()
        is_multilabel = task_name == "Task III"
        patient_id = self.get_selected_patient_id()
        if not patient_id:
            self.log("[INFO] Select a patient from the dropdown first.")
            return
        try:
            device = get_best_torch_device()
            self.hvdnet_model.to(device)
            self.hvdnet_model.eval()

            inference_result = self.build_patient_inference_result(patient_id, task_name, self.hvdnet_model, device)
            display_data = inference_result['data']
            class_names = inference_result['class_names']
            title_prefix = f"Inference | patient={patient_id} | task={task_name}"

            self.log(f"[INFERENCE] Task={task_name} | patient={patient_id} | segments={inference_result['num_segments']}")
            if is_multilabel:
                mean_prob_text = ", ".join(
                    f"{name}={prob * 100:.1f}%" for name, prob in zip(class_names, inference_result['mean_probabilities'])
                )
                patient_pred_labels = [
                    name for name, value in zip(class_names, inference_result['final_prediction']) if value > 0.5
                ]
                patient_true_labels = [
                    name for name, value in zip(class_names, inference_result['actual_truth']) if value > 0.5
                ]
                self.log(f" > Aggregated probabilities: {mean_prob_text}")
                self.log(f" > Final diagnosis: {', '.join(patient_pred_labels) if patient_pred_labels else 'Normal'}")
                self.log(f" > Actual truth: {', '.join(patient_true_labels) if patient_true_labels else 'Normal'}")
                self.log(f" > Exact match: {'YES' if inference_result['is_exact_match'] else 'NO'}")
            else:
                self.log(
                    f" > Final prediction: {class_names[int(inference_result['final_prediction'])]}"
                )
                self.log(
                    f" > Actual truth: {class_names[int(inference_result['actual_truth'])]}"
                )

            self.current_segment_idx = 0
            self.last_inference_result = {
                'patient_id': patient_id,
                'display_data': display_data,
                'title_prefix': title_prefix,
                'segment_results': inference_result['segment_results'],
                'full_attention': inference_result['full_attention'],
            }
            self.render_last_inference()
        except Exception as e:
            self.log(f"[ERROR] Inference attention failed: {str(e)}")

    def draw_confusion_matrix_on_plot(self, plot_item, cm, class_names, title=None):
        plot_item.clear()
        cm = np.asarray(cm, dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

        img = pg.ImageItem(cm_norm)
        img.setRect(QRectF(0, 0, cm.shape[1], cm.shape[0]))
        # Dark-to-cyan map avoids bright yellow, improving text readability.
        cm_colormap = pg.ColorMap(
            pos=np.array([0.0, 0.5, 1.0]),
            color=np.array([
                [12, 18, 45, 255],
                [35, 85, 145, 255],
                [90, 215, 225, 255],
            ], dtype=np.ubyte),
        )
        img.setLookupTable(cm_colormap.getLookupTable(0.0, 1.0, 256))
        plot_item.addItem(img)
        plot_item.setXRange(0, cm.shape[1], padding=0)
        plot_item.setYRange(0, cm.shape[0], padding=0)
        plot_item.invertY(True)
        if title is not None:
            plot_item.setTitle(title)

        x_ticks = [(i + 0.5, name) for i, name in enumerate(class_names)]
        y_ticks = [(i + 0.5, name) for i, name in enumerate(class_names)]
        plot_item.getAxis('bottom').setTicks([x_ticks])
        plot_item.getAxis('left').setTicks([y_ticks])

        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                text_color = (0, 0, 0) if cm_norm[r, c] >= 0.55 else (255, 255, 255)
                txt = pg.TextItem(f"{int(cm[r, c])}", color=text_color, anchor=(0.5, 0.5))
                txt.setPos(c + 0.5, r + 0.5)
                plot_item.addItem(txt)

    def draw_confusion_matrix(self, cm, class_names):
        self.cm_widget.clear()
        self.cm_plot = self.cm_widget.addPlot(row=0, col=0)
        self.draw_confusion_matrix_on_plot(self.cm_plot, cm, class_names, title="Confusion Matrix")

    def step_generate_confusion_matrix(self):
        if self.hvdnet_model is None:
            self.log("[INFO] Load or train a model first.")
            return

        task_name = self.loaded_model_task or self.task_selector.currentText()
        is_multilabel = task_name == "Task III"
        try:
            x_test, y_test, z_test, y_true, class_names = self.get_task_test_split_tensors(task_name)

            test_loader = DataLoader(
                TensorDataset(x_test, y_test, z_test, y_true),
                batch_size=64,
                shuffle=False,
            )

            device = get_best_torch_device()
            self.hvdnet_model.to(device)
            self.hvdnet_model.eval()

            preds = []
            targets = []
            with torch.no_grad():
                for bx, by, bz, bl in test_loader:
                    logits, _ = self.hvdnet_model(
                        move_tensor_to_device(bx, device),
                        move_tensor_to_device(by, device),
                        move_tensor_to_device(bz, device),
                    )
                    if is_multilabel:
                        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                        preds.append(pred)
                        targets.append(bl.cpu().numpy())
                    else:
                        pred = torch.argmax(logits, dim=1).cpu().numpy()
                        preds.extend(pred.tolist())
                        targets.extend(bl.cpu().numpy().tolist())

            if is_multilabel:
                targets_np = np.concatenate(targets, axis=0).astype(np.float32)
                preds_np = np.concatenate(preds, axis=0).astype(np.float32)
                self.cm_widget.clear()
                self.cm_plot = None
                per_class_accuracy = []
                per_class_precision = []
                per_class_recall = []
                per_class_f1 = []
                for idx, class_name in enumerate(class_names):
                    row = idx // 2
                    col = idx % 2
                    plot_item = self.cm_widget.addPlot(row=row, col=col)
                    class_targets = targets_np[:, idx].astype(int)
                    class_preds = preds_np[:, idx].astype(int)
                    cm = confusion_matrix(class_targets, class_preds, labels=[0, 1])
                    self.draw_confusion_matrix_on_plot(plot_item, cm, ["Absent", "Present"], title=class_name)

                    class_accuracy = accuracy_score(class_targets, class_preds)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        class_targets,
                        class_preds,
                        average='binary',
                        zero_division=0,
                    )
                    per_class_accuracy.append(class_accuracy)
                    per_class_precision.append(precision)
                    per_class_recall.append(recall)
                    per_class_f1.append(f1)

                self.log(f"[CONFUSION MATRIX] Task={task_name}")
                self.log(f" > Test samples: {len(targets_np)}")
                self.log(f" > Class order: {class_names}")
                self.log(
                    " > Macro metrics across labels: "
                    f"Accuracy={np.mean(per_class_accuracy) * 100:.2f}% | "
                    f"Precision={np.mean(per_class_precision) * 100:.2f}% | "
                    f"Recall={np.mean(per_class_recall) * 100:.2f}% | "
                    f"F1={np.mean(per_class_f1) * 100:.2f}%"
                )
                for idx, class_name in enumerate(class_names):
                    self.log(
                        f" > {class_name}: Acc={per_class_accuracy[idx] * 100:.2f}% | "
                        f"Prec={per_class_precision[idx] * 100:.2f}% | "
                        f"Rec={per_class_recall[idx] * 100:.2f}% | "
                        f"F1={per_class_f1[idx] * 100:.2f}%"
                    )
            else:
                cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
                self.draw_confusion_matrix(cm, class_names)

                accuracy = accuracy_score(targets, preds)
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    targets,
                    preds,
                    average='macro',
                    zero_division=0,
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    targets,
                    preds,
                    average='weighted',
                    zero_division=0,
                )

                self.log(f"[CONFUSION MATRIX] Task={task_name}")
                self.log(f" > Test samples: {len(targets)}")
                self.log(f" > Class order: {class_names}")
                self.log(
                    " > Metrics (macro): "
                    f"Accuracy={accuracy * 100:.2f}% | "
                    f"Precision={precision_macro * 100:.2f}% | "
                    f"Recall={recall_macro * 100:.2f}% | "
                    f"F1={f1_macro * 100:.2f}%"
                )
                self.log(
                    " > Metrics (weighted): "
                    f"Precision={precision_weighted * 100:.2f}% | "
                    f"Recall={recall_weighted * 100:.2f}% | "
                    f"F1={f1_weighted * 100:.2f}%"
                )
                self.log(f" > Matrix:\n{cm}")
        except Exception as e:
            self.log(f"[ERROR] Confusion matrix generation failed: {str(e)}")

    def get_mean_attention_for_class(self, model, test_dataset, target_class_idx, axis_name='X', num_samples=50, task_name="Task I"):
        model.eval()
        device = next(model.parameters()).device

        signal_key = axis_name.upper()
        collected_signals = []
        collected_attention = []
        count = 0

        with torch.no_grad():
            for i in range(len(test_dataset)):
                x, y, z, label = test_dataset[i]
                if task_name == "Task III":
                    if float(label[target_class_idx].item()) < 0.5:
                        continue
                else:
                    if int(label.item()) != int(target_class_idx):
                        continue

                x_input = move_tensor_to_device(x.unsqueeze(0), device)
                y_input = move_tensor_to_device(y.unsqueeze(0), device)
                z_input = move_tensor_to_device(z.unsqueeze(0), device)

                _, (attn_x, attn_y, attn_z) = model(x_input, y_input, z_input)

                if signal_key == 'X':
                    collected_signals.append(x.squeeze().cpu().numpy())
                    collected_attention.append(attn_x.squeeze().detach().cpu().numpy())
                elif signal_key == 'Y':
                    collected_signals.append(y.squeeze().cpu().numpy())
                    collected_attention.append(attn_y.squeeze().detach().cpu().numpy())
                else:
                    collected_signals.append(z.squeeze().cpu().numpy())
                    collected_attention.append(attn_z.squeeze().detach().cpu().numpy())

                count += 1
                if count >= num_samples:
                    break

        if count == 0:
            return None, None, 0

        mean_signal = np.mean(np.asarray(collected_signals), axis=0)
        mean_attention = np.mean(np.asarray(collected_attention), axis=0)
        scale = len(mean_signal) / max(len(mean_attention), 1)
        mean_attention_upsampled = scipy.ndimage.zoom(mean_attention, scale, order=1)
        if len(mean_attention_upsampled) < len(mean_signal):
            mean_attention_upsampled = np.pad(
                mean_attention_upsampled,
                (0, len(mean_signal) - len(mean_attention_upsampled)),
                mode='edge'
            )
        mean_attention_upsampled = mean_attention_upsampled[:len(mean_signal)]

        return mean_signal, mean_attention_upsampled, count

    def step_generate_class_attention_maps(self):
        if self.hvdnet_model is None:
            self.log("[INFO] Load or train a model first.")
            return

        task_name = self.loaded_model_task or self.task_selector.currentText()
        selected_class_name = self.class_attn_class_selector.currentText()
        try:
            num_samples = int(self.class_attn_samples_input.text().strip())
            if num_samples <= 0:
                raise ValueError
        except ValueError:
            self.log("[INFO] Invalid sample count; using default n=50.")
            num_samples = 50
            self.class_attn_samples_input.setText("50")

        try:
            x_test, y_test, z_test, y_true, class_names = self.get_task_test_split_tensors(task_name)
            test_dataset = TensorDataset(x_test, y_test, z_test, y_true)

            device = get_best_torch_device()
            self.hvdnet_model.to(device)
            self.hvdnet_model.eval()

            if selected_class_name not in class_names:
                self.refresh_class_attention_class_selector()
                selected_class_name = self.class_attn_class_selector.currentText()

            if selected_class_name not in class_names:
                self.log("[ERROR] Selected condition is not available for this task.")
                return

            target_class_idx = class_names.index(selected_class_name)

            self.class_attention_widget.clear()
            class_axis_plots = []
            for row_idx, axis_name in enumerate(("X", "Y", "Z")):
                mean_signal, mean_attn, count = self.get_mean_attention_for_class(
                    self.hvdnet_model,
                    test_dataset,
                    target_class_idx=target_class_idx,
                    axis_name=axis_name,
                    num_samples=num_samples,
                    task_name=task_name,
                )

                plot_item = self.class_attention_widget.addPlot(row=row_idx, col=0)
                class_axis_plots.append(plot_item)
                if mean_signal is None:
                    plot_item.setTitle(f"{selected_class_name} - {axis_name} Axis (n=0, no samples)")
                    plot_item.showGrid(x=True, y=True, alpha=0.25)
                    continue

                self.plot_attention_overlay(
                    mean_signal,
                    mean_attn,
                    plot_item,
                    axis_name,
                    f"{selected_class_name} (n={count})"
                )

            if class_axis_plots:
                master_plot = class_axis_plots[0]
                for plot_item in class_axis_plots[1:]:
                    plot_item.setXLink(master_plot)

            self.graph_tabs.setCurrentWidget(self.class_attention_tab)
            self.log(
                f"[CLASS ATTENTION] Task={task_name}, Condition={selected_class_name}, "
                f"Axes=X/Y/Z, Target n={num_samples}"
            )
            self.log(" > Generated mean signal + mean attention heatmaps for X, Y, Z of the selected condition.")
        except Exception as e:
            self.log(f"[ERROR] Class attention heatmap generation failed: {str(e)}")

    def apply_zero_phase_butterworth(self, signals_dict, fs, lowcut=1.0, highcut=30.0, order=6):
        nyquist = fs / 2.0
        if not (0 < lowcut < highcut < nyquist):
            raise ValueError(f"Invalid bandpass range: {lowcut}-{highcut} Hz for fs={fs} Hz")

        b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
        filtered = {}
        for name, values in signals_dict.items():
            filtered[name] = signal.filtfilt(b, a, np.asarray(values, dtype=float))
        return filtered

    def build_rpeak_segments(self, r_peaks, signal_length):
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
                    'end_peak_number': i + 3
                })
        return segments

    def build_threebeat_segments(self, r_peaks, signal_length, fs, min_bpm=40, max_bpm=120):
        """
        Build segments from r_peaks using 3-beat windows (P_i to P_{i+3}).
        Only keep segments whose average beat rate (computed from the 3-beat window)
        falls within [min_bpm, max_bpm]. Returns (segments, discarded_count).
        """
        r_peaks = np.asarray(r_peaks, dtype=int)
        segments = []
        discarded = 0
        if len(r_peaks) < 4:
            return segments, discarded

        min_seg_samples = int(np.round((3.0 * 60.0) / max_bpm * fs))  # 3 beats at max_bpm -> min segment length
        max_seg_samples = int(np.round((3.0 * 60.0) / min_bpm * fs))  # 3 beats at min_bpm -> max segment length

        for i in range(len(r_peaks) - 3):
            start_idx = int(r_peaks[i])
            end_idx = int(r_peaks[i + 3])
            seg_len = end_idx - start_idx
            if not (0 <= start_idx < end_idx <= signal_length):
                discarded += 1
                continue

            if min_seg_samples <= seg_len <= max_seg_samples:
                segments.append({
                    'segment_id': i,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_peak_number': i,
                    'end_peak_number': i + 3
                })
            else:
                discarded += 1

        return segments, int(discarded)

    def filter_peaks_by_physiology(self, peak_indices, fs, enabled=False, min_interval_sec=0.5, max_interval_sec=1.5):
        peak_indices = np.asarray(peak_indices, dtype=int)
        if len(peak_indices) <= 1 or not enabled:
            return peak_indices, 0

        # Less strict peak filtering:
        # - Only remove peaks that create intervals that are too short (likely duplicates/noise).
        # - Do NOT discard peaks that create long intervals (likely missed detections) because
        #   removing in that case removes valid data. This preserves valid peaks.

        sorted_unique = np.unique(np.sort(peak_indices)).astype(int)
        min_samples = int(np.round(min_interval_sec * fs))
        max_samples = int(np.round(max_interval_sec * fs))

        if len(sorted_unique) <= 1:
            return sorted_unique, 0

        keep = np.ones(len(sorted_unique), dtype=bool)

        # Iterate and remove obvious duplicates (too-short intervals).
        # Use an iterative pass because removing a peak changes neighboring intervals.
        removed = 0
        changed = True
        while changed:
            changed = False
            idxs = np.where(keep)[0]
            if len(idxs) <= 1:
                break
            positions = sorted_unique[idxs]
            intervals = np.diff(positions)
            for j, interval in enumerate(intervals):
                if interval < min_samples:
                    # violation between positions[j] and positions[j+1]
                    global_i = idxs[j]
                    global_j = idxs[j+1]

                    # Decide which of the two peaks to remove.
                    # Heuristic: remove the peak that is closer to its other neighbor (i.e., forms smaller adjacent interval),
                    # or prefer removing the later one if ambiguous.
                    left_gap = positions[j] - positions[j-1] if j-1 >= 0 else np.inf
                    right_gap = positions[j+2] - positions[j+1] if j+2 < len(positions) else np.inf

                    # Compare neighbor gaps to choose removal candidate
                    if left_gap < right_gap:
                        # remove earlier peak (global_i)
                        keep[global_i] = False
                    else:
                        # remove later peak (global_j)
                        keep[global_j] = False

                    removed += 1
                    changed = True
                    break
                # For intervals > max_samples we do NOT remove peaks here (too risky)

        accepted = sorted_unique[keep]
        return accepted.astype(int), int(removed)

    def on_view_mode_changed(self, _index):
        self.current_segment_idx = 0
        self.update_navigation_controls()
        self.plot_current_view()

    def show_previous_segment(self):
        inference_segment_mode = (
            self.graph_tabs.currentWidget() == self.inference_tab
            and self.last_inference_result is not None
            and self.inference_view_mode.currentIndex() == 1
        )

        if inference_segment_mode:
            self.current_segment_idx = max(0, self.current_segment_idx - 1)
            self.update_navigation_controls()
            self.render_last_inference()
            return

        if not self.segments:
            return
        self.current_segment_idx = max(0, self.current_segment_idx - 1)
        self.update_navigation_controls()
        self.plot_current_view()

    def show_next_segment(self):
        inference_segment_mode = (
            self.graph_tabs.currentWidget() == self.inference_tab
            and self.last_inference_result is not None
            and self.inference_view_mode.currentIndex() == 1
        )

        if inference_segment_mode:
            total_segments = len(self.last_inference_result.get('segment_results', []))
            if total_segments == 0:
                return
            self.current_segment_idx = min(total_segments - 1, self.current_segment_idx + 1)
            self.update_navigation_controls()
            self.render_last_inference()
            return

        if not self.segments:
            return
        self.current_segment_idx = min(len(self.segments) - 1, self.current_segment_idx + 1)
        self.update_navigation_controls()
        self.plot_current_view()

    def update_navigation_controls(self):
        segment_mode = self.view_mode.currentIndex() == 1
        has_segments = len(self.segments) > 0
        inference_segment_mode = (
            self.graph_tabs.currentWidget() == self.inference_tab
            and self.last_inference_result is not None
            and self.inference_view_mode.currentIndex() == 1
            and len(self.last_inference_result.get('segment_results', [])) > 0
        )
        total_inference_segments = len(self.last_inference_result.get('segment_results', [])) if self.last_inference_result else 0
        current_inference_segment = min(self.current_segment_idx + 1, max(total_inference_segments, 1)) if total_inference_segments else 0

        self.prev_segment_btn.setEnabled(
            (segment_mode and has_segments and self.current_segment_idx > 0)
            or (inference_segment_mode and self.current_segment_idx > 0)
        )
        self.next_segment_btn.setEnabled(
            (segment_mode and has_segments and self.current_segment_idx < len(self.segments) - 1)
            or (inference_segment_mode and self.current_segment_idx < total_inference_segments - 1)
        )

        if inference_segment_mode:
            self.segment_info_label.setText(
                f"Inference Segment: {current_inference_segment}/{total_inference_segments}"
            )
            return

        if not has_segments:
            self.segment_info_label.setText("Segment: N/A")
            return

        if segment_mode:
            current_seg_num = self.current_segment_idx + 1
            total_seg = len(self.segments)
            seg = self.segments[self.current_segment_idx]
            self.segment_info_label.setText(
                f"Segment: {current_seg_num}/{total_seg} (P{seg['start_peak_number']}-P{seg['end_peak_number']})"
            )
        else:
            self.segment_info_label.setText(f"Segments available: {len(self.segments)}")

    def plot_current_view(self, *_args):
        if not self.current_data:
            self.clear_plots()
            return

        fs = self.current_data['fs']
        stage_index = self.plot_stage.currentIndex()
        stage_name = self.plot_stage.currentText()

        if stage_index == 0:
            signals = self.current_data['signals']
        elif stage_index == 1:
            signals = self.current_data.get('filtered_signals', self.current_data['signals'])
        else:
            signals = self.current_data.get('filtered_signals', self.current_data['signals'])

        accx = signals['AccX']
        accy = signals['AccY']
        accz = signals['AccZ']
        ecg = signals['ECG']
        peak_indices = np.asarray(self.current_data.get('r_peaks_indices', []), dtype=int)
        peak_label = self.current_data.get('peak_label', 'R-peaks')
        peak_plot_axis = self.current_data.get('peak_plot_axis', 'ECG')

        segment_mode = self.view_mode.currentIndex() == 1 and len(self.segments) > 0

        if segment_mode and stage_index == 2 and self.preprocessed_segments:
            seg = self.preprocessed_segments[self.current_segment_idx]
            t = np.arange(800) / fs

            accx_v = seg['AccX'].reshape(-1)
            accy_v = seg['AccY'].reshape(-1)
            accz_v = seg['AccZ'].reshape(-1)
            ecg_v = seg['ECG_fixed']

            start_idx = seg['start_idx']
            end_idx = seg['end_idx']
            seg_mask = (peak_indices >= start_idx) & (peak_indices < end_idx)
            peak_indices_local = peak_indices[seg_mask] - start_idx
            peak_indices_local = peak_indices_local[peak_indices_local < 800]
            peak_times = peak_indices_local / fs
            if peak_plot_axis == 'AccZ':
                peak_vals = accz_v[peak_indices_local] if len(peak_indices_local) else np.array([])
            else:
                peak_vals = ecg_v[peak_indices_local] if len(peak_indices_local) else np.array([])

            title = (
                f"Patient {self.current_patient_id} - {stage_name}, "
                f"Segment {self.current_segment_idx + 1}/{len(self.preprocessed_segments)}"
            )
            show_peak_lines = True
        elif segment_mode:
            seg = self.segments[self.current_segment_idx]
            start_idx = seg['start_idx']
            end_idx = seg['end_idx']

            accx_v = accx[start_idx:end_idx]
            accy_v = accy[start_idx:end_idx]
            accz_v = accz[start_idx:end_idx]
            ecg_v = ecg[start_idx:end_idx]
            t = np.arange(end_idx - start_idx) / fs

            seg_mask = (peak_indices >= start_idx) & (peak_indices < end_idx)
            peak_indices_local = peak_indices[seg_mask] - start_idx
            peak_times = peak_indices_local / fs
            if peak_plot_axis == 'AccZ':
                peak_vals = accz_v[peak_indices_local] if len(peak_indices_local) else np.array([])
            else:
                peak_vals = ecg_v[peak_indices_local] if len(peak_indices_local) else np.array([])

            title = (
                f"Patient {self.current_patient_id} - {stage_name}, "
                f"Segment {self.current_segment_idx + 1}/{len(self.segments)}"
            )
            show_peak_lines = True
        else:
            n = self.current_data['signal_length']
            t = np.arange(n) / fs
            accx_v = accx
            accy_v = accy
            accz_v = accz
            ecg_v = ecg

            peak_times = peak_indices / fs if len(peak_indices) else np.array([])
            if len(peak_indices):
                if peak_plot_axis == 'AccZ':
                    peak_vals = accz[peak_indices]
                else:
                    peak_vals = ecg[peak_indices]
            else:
                peak_vals = np.array([])

            title = f"Patient {self.current_patient_id} - {stage_name}, Entire Signal"
            show_peak_lines = False

        self.clear_plots()

        self.accx_plot.setTitle(title)
        self.accx_plot.plot(t, accx_v, pen=pg.mkPen('#1f77b4', width=1))
        self.accy_plot.plot(t, accy_v, pen=pg.mkPen('#ff7f0e', width=1))
        self.accz_plot.plot(t, accz_v, pen=pg.mkPen('#2ca02c', width=1))
        self.ecg_plot.plot(t, ecg_v, pen=pg.mkPen('#d62728', width=1))

        if len(peak_times):
            peak_plot_target = self.accz_plot if peak_plot_axis == 'AccZ' else self.ecg_plot
            if show_peak_lines:
                for peak_time in peak_times:
                    peak_plot_target.addLine(x=float(peak_time), pen=pg.mkPen((255, 255, 0, 90), width=1))

            r_peak_scatter = pg.ScatterPlotItem(
                x=peak_times,
                y=peak_vals,
                pen=pg.mkPen((0, 0, 0), width=1),
                brush=pg.mkBrush(255, 255, 0),
                size=10,
                name=peak_label
            )
            peak_plot_target.addItem(r_peak_scatter)

        if len(t):
            initial_window_sec = min(10, t[-1])
            if initial_window_sec > 0:
                self.accx_plot.setXRange(0, initial_window_sec, padding=0.01)

    def clear_plots(self):
        self.accx_plot.clear()
        self.accy_plot.clear()
        self.accz_plot.clear()
        self.ecg_plot.clear()

    def clear_inference_plots(self):
        self.inf_accx_plot.clear()
        self.inf_accy_plot.clear()
        self.inf_accz_plot.clear()
        self.inf_ecg_plot.clear()

    def render_last_inference(self):
        if self.last_inference_result is None:
            return

        result = self.last_inference_result
        if self.inference_view_mode.currentIndex() == 0:
            self.plot_inference_full_signal(
                result['patient_id'],
                result['display_data'],
                result['title_prefix'],
                plot_targets=(self.inf_accx_plot, self.inf_accy_plot, self.inf_accz_plot, self.inf_ecg_plot),
                attention_maps=result['full_attention'],
            )
        else:
            segment_results = result['segment_results']
            if not segment_results:
                return
            self.current_segment_idx = min(self.current_segment_idx, len(segment_results) - 1)
            segment = segment_results[self.current_segment_idx]
            self.clear_inference_plots()
            segment_title = (
                f"Inference | Segment {segment['segment_idx'] + 1}/{len(segment_results)} | "
                f"{result['title_prefix']}"
            )
            self.plot_attention_overlay(segment['signal_x'], segment['attention_x'], self.inf_accx_plot, 'X', segment_title)
            self.plot_attention_overlay(segment['signal_y'], segment['attention_y'], self.inf_accy_plot, 'Y', segment_title)
            self.plot_attention_overlay(segment['signal_z'], segment['attention_z'], self.inf_accz_plot, 'Z', segment_title)
            self.inf_ecg_plot.clear()
            self.inf_ecg_plot.plot(np.arange(len(segment['signal_ecg'])), segment['signal_ecg'], pen=pg.mkPen('#d62728', width=1))
            self.inf_ecg_plot.setTitle(f"ECG | Segment {segment['segment_idx'] + 1}/{len(segment_results)}")

        self.graph_tabs.setCurrentWidget(self.inference_tab)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HVDMainWindow()
    window.show()
    sys.exit(app.exec_())