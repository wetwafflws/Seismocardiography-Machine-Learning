import sys
import json
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from pipeline import preprocess_scg, svmd_extract_modes, select_and_reconstruct_ao, extract_ao_peaks


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")


def get_original_fs(patient_id):
    if patient_id.startswith("UP-"):
        try:
            patient_number = int(patient_id.split("-")[1])
            if 22 <= patient_number <= 30:
                return 512
        except ValueError:
            pass
    return 256

class SCGAnalyzerApp(QMainWindow):
    def __init__(self, csv_path, json_path, fs=256):
        super().__init__()
        self.fs = fs
        self.window_duration = 10  # 10 seconds sliding window
        self.window_samples = self.window_duration * self.fs
        self.current_imfs = np.array([])
        self.current_mode_index = 0
        self.current_time_window = np.array([])
        
        self.load_data(csv_path, json_path)
        self.init_ui()
        self.update_plots(0) # Initialize with first segment

    def resolve_data_path(self, path_or_name, default_extension=None):
        if os.path.exists(path_or_name):
            return path_or_name

        if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
            return path_or_name

        candidate = os.path.join(DATA_DIR, path_or_name)
        if os.path.exists(candidate):
            return candidate

        if default_extension and not path_or_name.lower().endswith(default_extension.lower()):
            candidate = os.path.join(DATA_DIR, f"{path_or_name}{default_extension}")
            if os.path.exists(candidate):
                return candidate

        return candidate

    def load_data(self, csv_path, json_path):
        csv_path = self.resolve_data_path(csv_path)
        json_path = self.resolve_data_path(json_path)

        # Load CSV in the same comma-separated format used by machinelearning.py
        df = pd.read_csv(csv_path, sep=',')
        self.ecg_full = df['ECG'].values
        self.scg_full = df['AccZ'].values
        self.time_full = np.arange(len(self.scg_full)) / self.fs
        
        # Load JSON Ground Truth
        with open(json_path, 'r') as f:
            gt_data = json.load(f)
            
        self.r_peak_indices = []
        for t_str in gt_data.get("LARA_R_Peaks", []):
            hours, minutes, seconds = t_str.split(":")
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            idx = int(total_seconds * self.fs)
            if idx < len(self.time_full):
                self.r_peak_indices.append(idx)
        
        self.r_peak_indices = np.array(self.r_peak_indices)
        self.max_start_idx = len(self.scg_full) - self.window_samples

    def init_ui(self):
        self.setWindowTitle("SCG AO Peak Detection Pipeline")
        self.resize(1200, 900)
        
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Plotting Area
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)
        
        # 1. ECG Panel
        self.p_ecg = self.glw.addPlot(title="1. ECG & Ground Truth R-Peaks")
        self.curve_ecg = self.p_ecg.plot(pen='b')
        self.scatter_r_peaks = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
        self.p_ecg.addItem(self.scatter_r_peaks)
        self.glw.nextRow()
        
        # 2. Raw SCG (AccZ)
        self.p_raw = self.glw.addPlot(title="2. Raw SCG (AccZ)")
        self.curve_raw = self.p_raw.plot(pen='w')
        self.p_raw.setXLink(self.p_ecg)
        self.glw.nextRow()
        
        # 3. Preprocessed SCG
        self.p_pre = self.glw.addPlot(title="3. Preprocessed SCG (MTI Filtered)")
        self.curve_pre = self.p_pre.plot(pen='c')
        self.p_pre.setXLink(self.p_ecg)
        self.glw.nextRow()
        
        # 4. IMFs
        self.p_imf = self.glw.addPlot(title="4. SVMD Decomposed IMFs (Selected = Green, Ignored = Gray)")
        self.imf_curves = []
        self.p_imf.setXLink(self.p_ecg)
        self.glw.nextRow()

        mode_control_row = QHBoxLayout()
        mode_control_row.addWidget(QLabel("Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Mode 1"])
        self.mode_selector.currentIndexChanged.connect(self.on_mode_selected)
        mode_control_row.addWidget(self.mode_selector)
        mode_control_row.addStretch()
        layout.addLayout(mode_control_row)

        self.p_mode_detail = self.glw.addPlot(title="4b. Selected Decomposed Mode")
        self.p_mode_detail.setXLink(self.p_ecg)
        self.selected_mode_curve = self.p_mode_detail.plot(pen='y')
        self.glw.nextRow()
        
        # 5. Final AO Peaks
        self.p_final = self.glw.addPlot(title="5. Reconstructed Signal, Smoothed Envelope (7th Power) & AO Peaks")
        self.curve_ao_sig = self.p_final.plot(pen=(100, 100, 100)) # Reconstructed sig
        self.curve_env = self.p_final.plot(pen='g') # Envelope
        self.scatter_ao_peaks = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200))
        self.p_final.addItem(self.scatter_ao_peaks)
        self.p_final.setXLink(self.p_ecg)
        
        # Navigation Slider
        nav_layout = QHBoxLayout()
        self.lbl_time = QLabel("0s")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.max_start_idx)
        self.slider.valueChanged.connect(self.on_slider_moved)
        
        nav_layout.addWidget(QLabel("Slide 10s Window:"))
        nav_layout.addWidget(self.slider)
        nav_layout.addWidget(self.lbl_time)
        
        layout.addLayout(nav_layout)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def on_mode_selected(self, index):
        self.current_mode_index = index
        if hasattr(self, "current_imfs") and np.size(self.current_imfs) > 0 and 0 <= index < len(self.current_imfs):
            self.plot_selected_mode(self.current_imfs, self.current_time_window)

    def on_slider_moved(self, value):
        self.lbl_time.setText(f"{value / self.fs:.1f}s")
        self.update_plots(value)

    def update_plots(self, start_idx):
        end_idx = start_idx + self.window_samples
        t_win = self.time_full[start_idx:end_idx]
        
        # 1. Update ECG
        ecg_win = self.ecg_full[start_idx:end_idx]
        self.curve_ecg.setData(t_win, ecg_win)
        self.current_time_window = t_win
        
        # Find R-peaks within this window
        window_r_peaks = self.r_peak_indices[(self.r_peak_indices >= start_idx) & (self.r_peak_indices < end_idx)]
        self.scatter_r_peaks.setData(self.time_full[window_r_peaks], self.ecg_full[window_r_peaks])
        
        # 2. Update Raw SCG
        scg_win = self.scg_full[start_idx:end_idx]
        self.curve_raw.setData(t_win, scg_win)
        
        # 3. Preprocess
        pre_scg = preprocess_scg(scg_win)
        self.curve_pre.setData(t_win, pre_scg)
        
        # 4. SVMD Decomposition
        # Using 5 modes to keep UI responsive. Increase to mimic MATLAB exactly, but expect lag on slider drag.
        imfs = svmd_extract_modes(pre_scg, max_modes=5)
        self.current_imfs = imfs
        self.refresh_mode_selector(len(imfs))
        
        # Clear old IMF curves
        for c in self.imf_curves:
            self.p_imf.removeItem(c)
        self.imf_curves.clear()
        
        # Select modes and plot
        s_AO, _, selected_indices = select_and_reconstruct_ao(imfs)
        
        offset = 0
        for i, imf in enumerate(imfs):
            color = 'g' if i in selected_indices else (100, 100, 100)
            c = self.p_imf.plot(t_win, imf + offset, pen=color)
            self.imf_curves.append(c)
            offset -= (np.max(imf) - np.min(imf)) * 1.2 # Stack them visually

        self.plot_selected_mode(imfs, t_win)
            
        # 5. Extract AO Peaks
        env, ao_peak_idx_local = extract_ao_peaks(s_AO, self.fs)
        
        # Normalize for display purposes
        s_AO_disp = s_AO / np.max(np.abs(s_AO)) if np.max(np.abs(s_AO)) > 0 else s_AO
        env_disp = env / np.max(env) if np.max(env) > 0 else env
        
        self.curve_ao_sig.setData(t_win, s_AO_disp)
        self.curve_env.setData(t_win, env_disp)
        
        ao_times = t_win[ao_peak_idx_local]
        ao_vals = env_disp[ao_peak_idx_local]
        self.scatter_ao_peaks.setData(ao_times, ao_vals)

    def refresh_mode_selector(self, num_modes):
        if num_modes <= 0:
            num_modes = 1

        self.mode_selector.blockSignals(True)
        self.mode_selector.clear()
        self.mode_selector.addItems([f"Mode {i + 1}" for i in range(num_modes)])
        self.mode_selector.setCurrentIndex(min(self.current_mode_index, num_modes - 1))
        self.mode_selector.blockSignals(False)

    def plot_selected_mode(self, imfs, time_axis):
        self.p_mode_detail.clear()

        if imfs is None or len(imfs) == 0:
            self.selected_mode_curve = self.p_mode_detail.plot([], [], pen='y')
            self.p_mode_detail.setTitle("4b. Selected Decomposed Mode")
            return

        mode_index = min(self.mode_selector.currentIndex(), len(imfs) - 1)
        mode_signal = np.asarray(imfs[mode_index], dtype=float)
        mode_time = np.asarray(time_axis[:len(mode_signal)], dtype=float)

        self.selected_mode_curve = self.p_mode_detail.plot(mode_time, mode_signal, pen=pg.mkPen('#ffd166', width=2))
        self.p_mode_detail.setTitle(f"4b. Selected Decomposed Mode - Mode {mode_index + 1}")
        self.p_mode_detail.setLabel('left', f'Mode {mode_index + 1}')
        self.p_mode_detail.setLabel('bottom', 'Time (s)')
        self.p_mode_detail.showGrid(x=True, y=True, alpha=0.25)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    patient_id = "CP-01"
    csv_file = os.path.join(DATA_DIR, f"Cleaned_{patient_id}.csv")
    json_file = os.path.join(DATA_DIR, f"{patient_id}-ECG.json")
    fs = get_original_fs(patient_id)
    
    ex = SCGAnalyzerApp(csv_file, json_file, fs=fs)
    ex.show()
    sys.exit(app.exec_())