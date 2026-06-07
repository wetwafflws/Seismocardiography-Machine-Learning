import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QGroupBox, QProgressBar, QStatusBar, QFrame, QLineEdit,
    QMessageBox, QComboBox, QDateEdit, QTimeEdit, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QTime
from PyQt5.QtGui import QColor

import pyqtgraph as pg
pg.setConfigOption('background', '#1a1a2e')
pg.setConfigOption('foreground', '#e0e0e0')
pg.setConfigOptions(antialias=True)


# ─────────────────────────────────────────────
# SCP File Parser
# ─────────────────────────────────────────────

def read_scp_ecg(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 400:
        raise ValueError(f"File terlalu kecil: {len(data)} bytes")
    if b'SCPECG' not in data[:30]:
        raise ValueError("Bukan file SCP-ECG yang valid")

    b_interval_low  = data[190]
    b_interval_high = data[191]
    interval_us = b_interval_low | (b_interval_high << 8)

    if interval_us > 0:
        FS = round(1000000.0 / interval_us, 2)
    else:
        FS = 150.0

    DATA_START    = 196
    SECTION_6_END = 173 + 9024  # = 9197
    N_SAMPLES     = (SECTION_6_END - DATA_START) // 2
    actual_samples = min(N_SAMPLES, (len(data) - DATA_START) // 2)

    ecg_signal = np.zeros(actual_samples, dtype=float)

    for i in range(actual_samples):
        bo       = DATA_START + i * 2
        b0       = data[bo]
        b1       = data[bo + 1]
        b1_clean = b1 & 0x0F
        val      = (b1_clean << 8) | b0
        ecg_signal[i] = val

    baseline = np.median(ecg_signal)
    ecg_mv   = (ecg_signal - baseline) * (10.0 / 2048.0)

    return ecg_mv, FS


# ─────────────────────────────────────────────
# Worker Thread
# ─────────────────────────────────────────────

class ProcessingWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, scp_files):
        super().__init__()
        self.scp_files = scp_files

    def run(self):
        try:
            all_ecg = []
            fs_list = []

            total_files = len(self.scp_files)
            for i, fpath in enumerate(self.scp_files):
                self.progress.emit(int((i / total_files) * 80), f"Membaca file {i+1}/{total_files}...")
                ecg_mv, fs = read_scp_ecg(fpath)
                all_ecg.append(ecg_mv)
                fs_list.append(fs)

            self.progress.emit(90, "Menggabungkan sinyal...")
            full_ecg = np.concatenate(all_ecg)
            final_fs = np.mean(fs_list)
            n_total  = len(full_ecg)
            time_arr = np.arange(n_total) / final_fs

            self.progress.emit(100, "Selesai!")

            self.finished.emit({
                'ecg':        full_ecg,
                'time':       time_arr,
                'fs':         final_fs,
                'n_total':    n_total,
                'duration_s': n_total / final_fs,
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# ─────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────

class ECGHolterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scp_files        = []
        self.result           = None
        self.worker           = None
        self._region_updating = False
        self._view_window_s   = 10.0   # lebar tampilan aktif (5 atau 10 detik)
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("ECG Holter Converter & Visualizer")
        self.setMinimumSize(1280, 860)
        self.setStyleSheet(STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ══════════════════════════════════════
        # LEFT PANEL
        # ══════════════════════════════════════
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(6)

        title = QLabel("⚕ ECG Holter")
        title.setObjectName("appTitle")
        left_layout.addWidget(title)

        # ── 1. Informasi Subjek ──
        gb_subject = QGroupBox("1. Informasi Subjek")
        gb_subject.setObjectName("groupBox")
        subj_layout = QVBoxLayout(gb_subject)
        subj_layout.setSpacing(5)

        self.txt_inisial = QLineEdit()
        self.txt_inisial.setPlaceholderText("Inisial Subjek")
        self.txt_inisial.setMaxLength(10)
        self.txt_inisial.setObjectName("textInput")
        self.txt_inisial.textChanged.connect(self._validate_ready)

        self.txt_subject = QLineEdit()
        self.txt_subject.setPlaceholderText("Kode Subjek (Maks 15 Karakter)")
        self.txt_subject.setMaxLength(15)
        self.txt_subject.setObjectName("textInput")
        self.txt_subject.textChanged.connect(self._validate_ready)

        row_demog = QHBoxLayout()
        self.combo_gender = QComboBox()
        self.combo_gender.setObjectName("comboSegment")
        self.combo_gender.addItems(["Jenis Kelamin", "M", "F"])
        self.combo_gender.currentIndexChanged.connect(self._validate_ready)

        self.txt_usia = QLineEdit()
        self.txt_usia.setPlaceholderText("Usia")
        self.txt_usia.setMaxLength(3)
        self.txt_usia.setObjectName("textInput")
        self.txt_usia.textChanged.connect(self._validate_ready)
        row_demog.addWidget(self.combo_gender)
        row_demog.addWidget(self.txt_usia)

        subj_layout.addWidget(self.txt_inisial)
        subj_layout.addWidget(self.txt_subject)
        subj_layout.addLayout(row_demog)
        left_layout.addWidget(gb_subject)

        # ── 2. Timestamp Rekaman ──
        gb_time = QGroupBox("2. Waktu Mulai Rekaman")
        gb_time.setObjectName("groupBox")
        time_layout = QVBoxLayout(gb_time)
        time_layout.setSpacing(5)

        lbl_tgl = QLabel("Tanggal:")
        lbl_tgl.setObjectName("infoLabel")
        self.date_edit = QDateEdit()
        self.date_edit.setObjectName("textInput")
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setDisplayFormat("dd/MM/yyyy")

        lbl_jam = QLabel("Waktu (HH:MM:SS):")
        lbl_jam.setObjectName("infoLabel")
        self.time_edit = QTimeEdit()
        self.time_edit.setObjectName("textInput")
        self.time_edit.setDisplayFormat("HH:mm:ss")
        self.time_edit.setTime(QTime(0, 0, 0))

        time_layout.addWidget(lbl_tgl)
        time_layout.addWidget(self.date_edit)
        time_layout.addWidget(lbl_jam)
        time_layout.addWidget(self.time_edit)
        left_layout.addWidget(gb_time)

        # ── 3. Pilih File SCP ──
        gb_files = QGroupBox("3. Pilih File SCP")
        gb_files.setObjectName("groupBox")
        files_layout = QVBoxLayout(gb_files)

        self.file_list = QListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.setFixedHeight(150)
        files_layout.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("＋ Tambah File")
        self.btn_add.setObjectName("btnSecondary")
        self.btn_clear = QPushButton("✕ Hapus Semua")
        self.btn_clear.setObjectName("btnDanger")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_clear)
        files_layout.addLayout(btn_row)

        self.lbl_file_count = QLabel("0 file dipilih")
        self.lbl_file_count.setObjectName("infoLabel")
        files_layout.addWidget(self.lbl_file_count)
        left_layout.addWidget(gb_files)

        # ── 4. Proses ──
        self.btn_process = QPushButton("▶  PROSES DATA")
        self.btn_process.setObjectName("btnPrimary")
        self.btn_process.setEnabled(False)
        self.btn_process.setFixedHeight(44)
        left_layout.addWidget(self.btn_process)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("statusLabel")
        left_layout.addWidget(self.lbl_status)

        # ── 5. Simpan ──
        self.btn_save_csv = QPushButton("📁  SIMPAN KE CSV")
        self.btn_save_csv.setObjectName("btnPrimary")
        self.btn_save_csv.setFixedHeight(44)
        self.btn_save_csv.setEnabled(False)
        left_layout.addStretch()
        left_layout.addWidget(self.btn_save_csv)

        root.addWidget(left_panel)

        # ══════════════════════════════════════
        # RIGHT PANEL
        # ══════════════════════════════════════
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)

        # Plot Overview
        self.plot_full = pg.PlotWidget(title="Sinyal Keseluruhan (Overview)")
        self.plot_full.setLabel('left', 'Amplitudo', units='mV')
        self.plot_full.setLabel('bottom', 'Waktu', units='s')
        self.plot_full.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.plot_full, stretch=1)

        self.overview_region = pg.LinearRegionItem(
            values=[0, 10],
            brush=pg.mkBrush(color=(100, 100, 255, 50))
        )
        self.overview_region.setZValue(10)
        self.plot_full.addItem(self.overview_region)

        # Kontrol Navigasi + Toggle Lebar
        btn_bar = QHBoxLayout()

        self.btn_prev_seg = QPushButton("◀")
        self.btn_prev_seg.setObjectName("btnNav")
        self.btn_prev_seg.setFixedWidth(30)
        self.btn_prev_seg.setEnabled(False)

        self.combo_segment = QComboBox()
        self.combo_segment.setObjectName("comboSegment")
        self.combo_segment.setFixedWidth(200)
        self.combo_segment.setEnabled(False)

        self.btn_next_seg = QPushButton("▶")
        self.btn_next_seg.setObjectName("btnNav")
        self.btn_next_seg.setFixedWidth(30)
        self.btn_next_seg.setEnabled(False)

        # Toggle 5s / 10s
        self.btn_5s = QPushButton("5 dtk")
        self.btn_5s.setObjectName("btnToggleOff")
        self.btn_5s.setFixedWidth(52)
        self.btn_5s.setFixedHeight(26)
        self.btn_5s.setCheckable(True)

        self.btn_10s = QPushButton("10 dtk")
        self.btn_10s.setObjectName("btnToggleOn")
        self.btn_10s.setFixedWidth(52)
        self.btn_10s.setFixedHeight(26)
        self.btn_10s.setCheckable(True)
        self.btn_10s.setChecked(True)

        self._toggle_group = QButtonGroup(self)
        self._toggle_group.setExclusive(True)
        self._toggle_group.addButton(self.btn_5s)
        self._toggle_group.addButton(self.btn_10s)

        btn_bar.addStretch()
        btn_bar.addWidget(QLabel("Segmen:"))
        btn_bar.addWidget(self.btn_prev_seg)
        btn_bar.addWidget(self.combo_segment)
        btn_bar.addWidget(self.btn_next_seg)
        btn_bar.addSpacing(16)
        btn_bar.addWidget(QLabel("Tampil:"))
        btn_bar.addWidget(self.btn_5s)
        btn_bar.addWidget(self.btn_10s)
        btn_bar.addStretch()
        right_layout.addLayout(btn_bar)

        # Plot Segmen — stretch dikurangi agar tidak terlalu tinggi
        self.plot_segment = pg.PlotWidget(title="Detail Segmen")
        self.plot_segment.setLabel('left', 'Amplitudo', units='mV')
        self.plot_segment.setLabel('bottom', 'Waktu', units='s')
        self.plot_segment.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.plot_segment, stretch=1)   # stretch=1 (sebelumnya 2)

        root.addWidget(right_panel, stretch=1)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # ── Connections ──
        self.btn_add.clicked.connect(self._add_files)
        self.btn_clear.clicked.connect(self._clear_files)
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_save_csv.clicked.connect(self._save_to_csv)

        self.combo_segment.currentIndexChanged.connect(self._go_to_segment)
        self.btn_prev_seg.clicked.connect(
            lambda: self.combo_segment.setCurrentIndex(
                max(0, self.combo_segment.currentIndex() - 1)
            )
        )
        self.btn_next_seg.clicked.connect(
            lambda: self.combo_segment.setCurrentIndex(
                min(self.combo_segment.count() - 1, self.combo_segment.currentIndex() + 1)
            )
        )

        self.btn_5s.clicked.connect(lambda: self._set_view_window(5.0))
        self.btn_10s.clicked.connect(lambda: self._set_view_window(10.0))

        self.overview_region.sigRegionChanged.connect(self._update_segment_from_region)
        self.plot_segment.sigXRangeChanged.connect(self._update_region_from_segment)

    # ──────────────────────────────────────────
    # Validasi
    # ──────────────────────────────────────────

    def _validate_ready(self):
        inisial = self.txt_inisial.text().strip()
        kode    = self.txt_subject.text().strip()
        gender  = self.combo_gender.currentIndex() > 0
        usia    = self.txt_usia.text().strip().isdigit()
        files   = len(self.scp_files) > 0
        self.btn_process.setEnabled(bool(inisial) and bool(kode) and gender and usia and files)

    # ──────────────────────────────────────────
    # File Management
    # ──────────────────────────────────────────

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Pilih File SCP", "", "SCP Files (*.SCP *.scp);;All Files (*)"
        )
        if not paths:
            return
        paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or '0'))
        for p in paths:
            if p not in self.scp_files:
                self.scp_files.append(p)
                item = QListWidgetItem(f"  {os.path.basename(p)}")
                item.setForeground(QColor('#a0c4ff'))
                self.file_list.addItem(item)
        self._update_file_count()
        self._validate_ready()

    def _clear_files(self):
        self.scp_files.clear()
        self.file_list.clear()
        self._update_file_count()
        self._validate_ready()

    def _update_file_count(self):
        self.lbl_file_count.setText(f"{len(self.scp_files)} file dipilih")

    # ──────────────────────────────────────────
    # Processing
    # ──────────────────────────────────────────

    def _start_processing(self):
        self.btn_process.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.worker = ProcessingWorker(self.scp_files)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, value, msg):
        self.progress_bar.setValue(value)
        self.lbl_status.setText(msg)

    def _on_error(self, msg):
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", msg)

    def _on_finished(self, result):
        self.result = result
        self.btn_process.setEnabled(True)
        self.btn_save_csv.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("✅ Analisis Selesai!")
        self._plot_ecg(result)

    # ──────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────

    def _plot_ecg(self, result):
        ecg = result['ecg']
        t   = result['time']

        self.plot_full.clear()
        step_ov = max(1, len(ecg) // 10000)
        self.plot_full.plot(
            t[::step_ov], ecg[::step_ov],
            pen=pg.mkPen(color='#4fc3f7', width=1)
        )
        self.plot_full.addItem(self.overview_region)

        self.plot_segment.clear()
        self.plot_segment.plot(t, ecg, pen=pg.mkPen(color='#4fc3f7', width=1))

        duration     = result['duration_s']
        num_segments = int(np.ceil(duration / 10.0))

        self.combo_segment.blockSignals(True)
        self.combo_segment.clear()
        for i in range(num_segments):
            start = i * 10
            end   = min((i + 1) * 10, duration)
            self.combo_segment.addItem(f"Segmen {i+1} ({start}s – {end:.0f}s)")
        self.combo_segment.setEnabled(True)
        self.combo_segment.blockSignals(False)

        self.combo_segment.setCurrentIndex(0)
        self._go_to_segment(0)

        self.statusBar.showMessage(
            f"Fs: {result['fs']:.1f} Hz  |  "
            f"Sampel: {result['n_total']}  |  "
            f"Durasi: {duration:.1f} s"
        )

    # ──────────────────────────────────────────
    # Navigasi & Toggle Lebar
    # ──────────────────────────────────────────

    def _set_view_window(self, seconds):
        self._view_window_s = seconds
        # Refresh tampilan di segmen aktif
        idx = self.combo_segment.currentIndex()
        if idx >= 0 and self.result:
            self._go_to_segment(idx)

    def _go_to_segment(self, index):
        if index < 0 or not self.result:
            return
        # Tengahkan jendela tampilan di tengah segmen 10-detik
        seg_center = index * 10.0 + 5.0
        half       = self._view_window_s / 2.0
        start_t    = max(0, seg_center - half)
        end_t      = start_t + self._view_window_s
        self.plot_segment.setXRange(start_t, end_t, padding=0)
        self.btn_prev_seg.setEnabled(index > 0)
        self.btn_next_seg.setEnabled(index < self.combo_segment.count() - 1)

    def _update_segment_from_region(self):
        if self._region_updating:
            return
        self._region_updating = True
        minX, maxX = self.overview_region.getRegion()
        self.plot_segment.setXRange(minX, maxX, padding=0)
        self._region_updating = False

    def _update_region_from_segment(self):
        if self._region_updating:
            return
        self._region_updating = True
        rng = self.plot_segment.viewRange()[0]
        self.overview_region.setRegion(rng)
        self._region_updating = False

    # ──────────────────────────────────────────
    # Simpan CSV
    # ──────────────────────────────────────────

    def _build_start_datetime(self):
        """Gabungkan QDateEdit + QTimeEdit jadi objek datetime Python."""
        qd = self.date_edit.date()
        qt = self.time_edit.time()
        return datetime(qd.year(), qd.month(), qd.day(),
                        qt.hour(), qt.minute(), qt.second())

    def _save_to_csv(self):
        if not self.result:
            return

        kode         = self.txt_subject.text().strip()
        default_name = f"ecg_{kode}.csv"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Simpan Sinyal ECG (CSV)", default_name, "CSV Files (*.csv)"
        )
        if not filepath:
            return

        self.lbl_status.setText("Menyimpan ke CSV...")
        QApplication.processEvents()

        try:
            start_dt  = self._build_start_datetime()
            time_s    = self.result['time']

            # Hitung datetime absolut per sampel
            dt_absolute = [
                (start_dt + timedelta(seconds=float(t))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                for t in time_s
            ]

            df = pd.DataFrame({
                'inisial':           self.txt_inisial.text().strip(),
                'kode_subjek':       kode,
                'jenis_kelamin':     self.combo_gender.currentText(),
                'usia':              int(self.txt_usia.text().strip()),
                'tanggal_rekam':     start_dt.strftime("%Y-%m-%d"),
                'sample_index':      np.arange(self.result['n_total']),
                'time_s':            time_s,
                'datetime_absolute': dt_absolute,
                'ecg_mv':            self.result['ecg'],
            })

            df.to_csv(filepath, index=False)
            self.lbl_status.setText("✅ CSV Berhasil Disimpan!")
            QMessageBox.information(self, "Sukses", f"Data berhasil disimpan ke:\n{filepath}")

        except Exception as e:
            self.lbl_status.setText("❌ Gagal menyimpan CSV.")
            QMessageBox.critical(self, "Error", f"Gagal menyimpan file:\n{str(e)}")


# ─────────────────────────────────────────────
# Stylesheet
# ─────────────────────────────────────────────

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI';
    font-size: 13px;
}
#leftPanel {
    background-color: #161b22;
    border-radius: 12px;
    border: 1px solid #30363d;
}
#rightPanel { background-color: #0d1117; }
#appTitle { font-size: 22px; font-weight: bold; color: #4fc3f7; }
QGroupBox#groupBox {
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #8b949e;
}
#fileList {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    color: #a0c4ff;
}
#textInput, QDateEdit#textInput, QTimeEdit#textInput {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #c9d1d9;
    padding: 5px;
}
QDateEdit#textInput::drop-down, QTimeEdit#textInput::drop-down {
    border: none;
    width: 20px;
}
QPushButton#btnPrimary {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}
QPushButton#btnPrimary:disabled { background: #21262d; color: #484f58; }
QPushButton#btnSecondary, QPushButton#btnDanger, QPushButton#btnNav {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px;
}
QPushButton#btnToggleOn {
    background-color: #1f6feb;
    color: white;
    border: none;
    border-radius: 5px;
    font-size: 11px;
    font-weight: bold;
}
QPushButton#btnToggleOff {
    background-color: #21262d;
    color: #8b949e;
    border: 1px solid #30363d;
    border-radius: 5px;
    font-size: 11px;
}
QPushButton#btnToggleOff:checked, QPushButton#btnToggleOn:checked {
    background-color: #1f6feb;
    color: white;
    border: none;
}
QPushButton#btnToggleOff:!checked {
    background-color: #21262d;
    color: #8b949e;
    border: 1px solid #30363d;
}
#comboSegment {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
}
QProgressBar#progressBar {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    text-align: center;
}
QProgressBar#progressBar::chunk { background: #1f6feb; }
#infoLabel { color: #8b949e; font-size: 11px; }
#statusLabel { color: #8b949e; font-size: 11px; }
"""

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ECGHolterApp()
    window.show()
    sys.exit(app.exec_())