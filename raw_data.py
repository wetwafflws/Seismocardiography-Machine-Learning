"""
SCG / PPG Beat Visualizer
Reads binary packets from STM32 over USB CDC and displays:
  - SCG (X, Y, Z) at 256 Hz — rolling waveform
  - PPG beat events — overlaid markers + BPM readout

Packet format:
  SCG  [0xAA][0x01][ts:4B LE][x:2B][y:2B][z:2B][chk:1B]  = 13 bytes
  BEAT [0xAA][0x02][ts:4B LE][chk:1B]                     = 7 bytes
"""

import sys
import struct
import csv
import serial
import serial.tools.list_ports
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSplitter, QFrame, QSizePolicy, QCheckBox,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

import pyqtgraph as pg
import numpy as np
from scipy.signal import butter, sosfilt

# ─── Protocol ────────────────────────────────────────────────────────────────

MAGIC        = 0xAA
TYPE_SCG     = 0x01
TYPE_BEAT    = 0x02
SCG_PKT_LEN  = 13
BEAT_PKT_LEN = 7

def xor_checksum(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
    return c

def parse_packets(buf: bytearray):
    """
    Consume as many complete packets from buf as possible.
    Returns (scg_samples, beat_timestamps, remaining_buf)
      scg_samples      : list of (timestamp_ms, x, y, z)
      beat_timestamps  : list of timestamp_ms
    """
    scg_samples = []
    beat_timestamps = []

    while len(buf) >= BEAT_PKT_LEN:
        # Re-sync: scan for magic byte
        if buf[0] != MAGIC:
            buf.pop(0)
            continue

        if len(buf) < 2:
            break

        pkt_type = buf[1]

        if pkt_type == TYPE_SCG:
            if len(buf) < SCG_PKT_LEN:
                break
            pkt = bytes(buf[:SCG_PKT_LEN])
            chk = xor_checksum(pkt[:-1])
            if chk == pkt[-1]:
                ts, x, y, z = struct.unpack_from('<Ihhh', pkt, 2)
                scg_samples.append((ts, x, y, z))
            else:
                buf.pop(0)
                continue
            del buf[:SCG_PKT_LEN]

        elif pkt_type == TYPE_BEAT:
            if len(buf) < BEAT_PKT_LEN:
                break
            pkt = bytes(buf[:BEAT_PKT_LEN])
            chk = xor_checksum(pkt[:-1])
            if chk == pkt[-1]:
                (ts,) = struct.unpack_from('<I', pkt, 2)
                beat_timestamps.append(ts)
            else:
                buf.pop(0)
                continue
            del buf[:BEAT_PKT_LEN]

        else:
            # Unknown type — drop magic byte and re-sync
            buf.pop(0)

    return scg_samples, beat_timestamps, buf


# ─── Serial Reader Thread ─────────────────────────────────────────────────────

class SerialReader(QObject):
    data_ready  = pyqtSignal(list, list)   # (scg_samples, beat_timestamps)
    error       = pyqtSignal(str)

    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port   = port
        self.baud   = baud
        self._running = False
        self._ser   = None
        self._buf   = bytearray()

    def start(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.02)
            self._running = True
        except serial.SerialException as e:
            self.error.emit(str(e))
            return
        self._loop()

    def stop(self):
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _loop(self):
        while self._running:
            try:
                chunk = self._ser.read(256)
                if chunk:
                    self._buf.extend(chunk)
                    scg, beats, self._buf = parse_packets(self._buf)
                    if scg or beats:
                        self.data_ready.emit(scg, beats)
            except serial.SerialException as e:
                self.error.emit(str(e))
                break


class ReaderThread(QThread):
    def __init__(self, reader: SerialReader):
        super().__init__()
        self._reader = reader

    def run(self):
        self._reader.start()

    def stop(self):
        self._reader.stop()
        self.quit()
        self.wait()


# ─── Styling ──────────────────────────────────────────────────────────────────

BG          = "#0d0f14"
BG_PANEL    = "#13161e"
BG_CARD     = "#1a1e28"
BORDER      = "#252a38"
ACCENT      = "#00e5ff"       # cyan
ACCENT2     = "#ff4757"       # red for beat
GREEN       = "#2ed573"
MUTED       = "#4a5068"
TEXT        = "#e8eaf0"
TEXT_DIM    = "#6b7494"

COLORS_SCG  = [ACCENT, "#a78bfa", GREEN]   # X, Y, Z
LABEL_SCG   = ["X", "Y", "Z"]

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 11px;
}}
QComboBox {{
    background-color: {BG_CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 140px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {TEXT};
    selection-background-color: {BORDER};
}}
QPushButton {{
    background-color: {BG_CARD};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 5px 16px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: {BG};
}}
QPushButton:disabled {{
    color: {MUTED};
    border-color: {MUTED};
}}
QPushButton#stop_btn {{
    color: {ACCENT2};
    border-color: {ACCENT2};
}}
QPushButton#stop_btn:hover {{
    background-color: {ACCENT2};
    color: {BG};
}}
QLabel#stat_value {{
    color: {ACCENT};
    font-size: 28px;
    font-weight: bold;
}}
QLabel#stat_label {{
    color: {TEXT_DIM};
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
}}
QLabel#bpm_value {{
    color: {ACCENT2};
    font-size: 42px;
    font-weight: bold;
}}
QLabel#status_ok  {{ color: {GREEN};  }}
QLabel#status_err {{ color: {ACCENT2}; }}
QFrame#card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QSplitter::handle {{
    background-color: {BORDER};
    width: 1px;
}}
"""

def make_plot_widget(title: str, unit: str = "") -> pg.PlotWidget:
    pw = pg.PlotWidget()
    pw.setBackground(BG_PANEL)
    pw.showGrid(x=False, y=True, alpha=0.08)
    pw.getAxis('left').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('bottom').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('left').setPen(pg.mkPen(BORDER))
    pw.getAxis('bottom').setPen(pg.mkPen(BORDER))
    title_item = pg.LabelItem(
        f'<span style="color:{TEXT_DIM};font-size:10px;letter-spacing:2px">{title}</span>'
    )
    pw.addItem(title_item)
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=True)
    return pw


# ─── Main Window ──────────────────────────────────────────────────────────────

WINDOW_SECS  = 5          # seconds of history shown
SAMPLE_RATE  = 256        # Hz
WINDOW_N     = WINDOW_SECS * SAMPLE_RATE   # samples in rolling window
BPM_HISTORY  = 8          # beats used for BPM average
MAX_PLOT_POINTS = 2000     # cap visible samples per curve for faster UI redraw
BPF_LOW_HZ = 0.5
BPF_HIGH_HZ = 50.0

# GY-61 (ADXL335) analog conversion from MCU ADC counts to g.
# Firmware currently packs ADC samples as int16, so we unwrap to uint16 first.
ADC_FULL_SCALE_COUNTS = 65535.0
ADC_ZERO_G_COUNTS = ADC_FULL_SCALE_COUNTS / 2.0
ADC_VREF = 3.3
ADXL335_SENSITIVITY_V_PER_G = 0.3
ADC_COUNTS_PER_G = (ADXL335_SENSITIVITY_V_PER_G / ADC_VREF) * ADC_FULL_SCALE_COUNTS


def raw_packet_int16_to_adc_counts(raw_value: int) -> float:
    return float(raw_value & 0xFFFF)


def adc_counts_to_g(adc_counts: float, zero_g_counts: float = ADC_ZERO_G_COUNTS) -> float:
    return (adc_counts - zero_g_counts) / ADC_COUNTS_PER_G

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCG · PPG Monitor")
        self.resize(1280, 820)

        # Data buffers
        self._scg_x = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
        self._scg_y = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
        self._scg_z = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
        self._scg_ts = deque([0] * WINDOW_N, maxlen=WINDOW_N)
        self._beat_ts: list[int] = []          # all beat timestamps (ms)
        self._beat_intervals: deque = deque(maxlen=BPM_HISTORY)
        self._last_beat_ts: int | None = None
        self._sample_count = 0
        self._t_axis = np.linspace(-WINDOW_SECS, 0, WINDOW_N)
        self._plot_dirty = False

        # Serial state
        self._thread: ReaderThread | None = None
        self._reader: SerialReader | None = None

        # Recording state
        self._is_recording = False
        self._csv_file = None
        self._csv_writer = None
        self._record_path = ""
        self._record_samples = 0
        self._record_first_ts = None
        self._record_last_ts = None
        self._record_elapsed_secs = 0

        # Optional display/data filter
        self._filter_enabled = False
        self._bpf_sos = butter(2, [BPF_LOW_HZ, BPF_HIGH_HZ], btype='bandpass', fs=SAMPLE_RATE, output='sos')
        self._reset_filter_state()

        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._update_record_timer)

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

        # Keep UI refresh cadence fixed rather than redrawing on every serial callback.
        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._on_plot_timer)
        self._plot_timer.start(33)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # Left: plots
        plot_widget = self._build_plots()

        # Right: sidebar
        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(220)

        root_layout.addWidget(plot_widget, stretch=1)
        root_layout.addWidget(sidebar)

    def _build_plots(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Title bar
        title_row = QHBoxLayout()
        title_lbl = QLabel("SEISMOCARDIOGRAM")
        title_lbl.setStyleSheet(f"color:{ACCENT};font-size:13px;font-weight:bold;letter-spacing:3px;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()

        # Legend
        for i, (label, color) in enumerate(zip(LABEL_SCG, COLORS_SCG)):
            dot = QLabel(f"● {label}")
            dot.setStyleSheet(f"color:{color};font-size:11px;margin-left:8px;")
            title_row.addWidget(dot)

        layout.addLayout(title_row)

        # Three stacked plot widgets
        self._plots: list[pg.PlotWidget] = []
        self._curves: list[pg.PlotDataItem] = []
        self._beat_lines: list[list[pg.InfiniteLine]] = [[], [], []]

        axes = ["X AXIS", "Y AXIS", "Z AXIS"]
        for i, (axis_name, color) in enumerate(zip(axes, COLORS_SCG)):
            pw = make_plot_widget(axis_name)
            pw.setXRange(-WINDOW_SECS, 0, padding=0)
            pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            curve = pw.plot(
                self._t_axis,
                np.zeros(WINDOW_N),
                pen=pg.mkPen(color, width=1.5, cosmetic=True)
            )
            self._plots.append(pw)
            self._curves.append(curve)
            layout.addWidget(pw)

        # Last beat segment plot (between the latest two beats)
        self._segment_plot = make_plot_widget("LAST BEAT SEGMENT")
        self._segment_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._segment_plot.getAxis('bottom').setLabel('time (s)')
        self._segment_plot.showGrid(x=True, y=True, alpha=0.08)
        self._segment_curves: list[pg.PlotDataItem] = []
        for color in COLORS_SCG:
            self._segment_curves.append(
                self._segment_plot.plot([], [], pen=pg.mkPen(color, width=1.5, cosmetic=True))
            )
        layout.addWidget(self._segment_plot)

        return container

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── BPM card ──────────────────────────────────────────────────────────
        bpm_card = QFrame()
        bpm_card.setObjectName("card")
        bpm_layout = QVBoxLayout(bpm_card)
        bpm_layout.setContentsMargins(16, 16, 16, 16)
        bpm_layout.setSpacing(2)

        bpm_title = QLabel("HEART RATE")
        bpm_title.setObjectName("stat_label")
        self._bpm_label = QLabel("--")
        self._bpm_label.setObjectName("bpm_value")
        self._bpm_label.setAlignment(Qt.AlignCenter)
        bpm_unit = QLabel("BPM")
        bpm_unit.setObjectName("stat_label")
        bpm_unit.setAlignment(Qt.AlignCenter)

        bpm_layout.addWidget(bpm_title, alignment=Qt.AlignCenter)
        bpm_layout.addWidget(self._bpm_label)
        bpm_layout.addWidget(bpm_unit)
        layout.addWidget(bpm_card)

        # ── Stats card ────────────────────────────────────────────────────────
        stats_card = QFrame()
        stats_card.setObjectName("card")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(16, 14, 16, 14)
        stats_layout.setSpacing(10)

        self._stat_beats  = self._make_stat(stats_layout, "BEATS TOTAL",  "0")
        self._stat_rate   = self._make_stat(stats_layout, "SAMPLE RATE",  "-- Hz")
        self._stat_lost   = self._make_stat(stats_layout, "PARSE ERRORS", "0")
        layout.addWidget(stats_card)

        # ── Connection card ───────────────────────────────────────────────────
        conn_card = QFrame()
        conn_card.setObjectName("card")
        conn_layout = QVBoxLayout(conn_card)
        conn_layout.setContentsMargins(16, 14, 16, 14)
        conn_layout.setSpacing(8)

        port_lbl = QLabel("PORT")
        port_lbl.setObjectName("stat_label")
        self._port_combo = QComboBox()
        self._refresh_ports()

        refresh_btn = QPushButton("↻  REFRESH")
        refresh_btn.clicked.connect(self._refresh_ports)

        self._connect_btn = QPushButton("CONNECT")
        self._connect_btn.clicked.connect(self._on_connect)

        self._stop_btn = QPushButton("DISCONNECT")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.clicked.connect(self._on_disconnect)
        self._stop_btn.setEnabled(False)

        self._status_lbl = QLabel("IDLE")
        self._status_lbl.setObjectName("status_err")
        self._status_lbl.setAlignment(Qt.AlignCenter)

        conn_layout.addWidget(port_lbl)
        conn_layout.addWidget(self._port_combo)
        conn_layout.addWidget(refresh_btn)
        conn_layout.addWidget(self._connect_btn)
        conn_layout.addWidget(self._stop_btn)
        conn_layout.addWidget(self._status_lbl)
        layout.addWidget(conn_card)

        layout.addStretch()

        # ── Data collection controls ─────────────────────────────────────────
        self._save_start_btn = QPushButton("START SAVE")
        self._save_start_btn.clicked.connect(self._start_save)
        layout.addWidget(self._save_start_btn)

        self._save_stop_btn = QPushButton("STOP SAVE")
        self._save_stop_btn.clicked.connect(self._stop_save)
        self._save_stop_btn.setEnabled(False)
        layout.addWidget(self._save_stop_btn)

        self._record_timer_lbl = QLabel("REC TIME: 00:00")
        self._record_timer_lbl.setObjectName("stat_label")
        self._record_timer_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._record_timer_lbl)

        self._filter_checkbox = QCheckBox("BANDPASS 0.5–50 Hz")
        self._filter_checkbox.setChecked(False)
        self._filter_checkbox.toggled.connect(self._on_filter_toggled)
        layout.addWidget(self._filter_checkbox)

        # ── Clear button ──────────────────────────────────────────────────────
        clear_btn = QPushButton("CLEAR PLOTS")
        clear_btn.clicked.connect(self._clear_data)
        layout.addWidget(clear_btn)

        return sidebar

    def _make_stat(self, parent_layout, label_text: str, value_text: str):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("stat_label")
        val = QLabel(value_text)
        val.setObjectName("stat_value")
        val.setStyleSheet(f"color:{TEXT};font-size:13px;font-weight:bold;")
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    # ── Port management ───────────────────────────────────────────────────────

    def _refresh_ports(self):
        self._port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self._port_combo.addItem(f"{p.device}  {p.description[:28]}", userData=p.device)
        if not ports:
            self._port_combo.addItem("No ports found")

    # ── Connect / Disconnect ──────────────────────────────────────────────────

    def _on_connect(self):
        port = self._port_combo.currentData()
        if not port:
            return

        self._reader = SerialReader(port)
        self._reader.data_ready.connect(self._on_data)
        self._reader.error.connect(self._on_serial_error)

        self._thread = ReaderThread(self._reader)
        self._thread.start()

        self._connect_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText("CONNECTED")
        self._status_lbl.setObjectName("status_ok")
        self._status_lbl.setStyleSheet(f"color:{GREEN};")

        # Rate estimator timer
        self._rate_timer = QTimer()
        self._rate_timer.timeout.connect(self._update_rate)
        self._rate_timer.start(1000)
        self._rate_count = 0

    def _on_disconnect(self):
        if self._is_recording:
            self._stop_save()

        if self._thread:
            self._thread.stop()
            self._thread = None
        self._reader = None
        self._connect_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText("DISCONNECTED")
        self._status_lbl.setStyleSheet(f"color:{ACCENT2};")
        if hasattr(self, '_rate_timer'):
            self._rate_timer.stop()

    def _on_serial_error(self, msg: str):
        self._status_lbl.setText(f"ERR: {msg[:24]}")
        self._status_lbl.setStyleSheet(f"color:{ACCENT2};")
        self._on_disconnect()

    def _reset_filter_state(self):
        n_sections = self._bpf_sos.shape[0]
        self._zi_x = np.zeros((n_sections, 2), dtype=np.float64)
        self._zi_y = np.zeros((n_sections, 2), dtype=np.float64)
        self._zi_z = np.zeros((n_sections, 2), dtype=np.float64)

    def _apply_filter_sample(self, x_val: float, y_val: float, z_val: float) -> tuple[float, float, float]:
        x_out, self._zi_x = sosfilt(self._bpf_sos, np.array([x_val], dtype=np.float64), zi=self._zi_x)
        y_out, self._zi_y = sosfilt(self._bpf_sos, np.array([y_val], dtype=np.float64), zi=self._zi_y)
        z_out, self._zi_z = sosfilt(self._bpf_sos, np.array([z_val], dtype=np.float64), zi=self._zi_z)
        return float(x_out[0]), float(y_out[0]), float(z_out[0])

    def _on_filter_toggled(self, checked: bool):
        self._filter_enabled = checked
        self._reset_filter_state()

    # ── Data ingestion ────────────────────────────────────────────────────────

    def _on_data(self, scg_samples: list, beat_timestamps: list):
        for ts, x, y, z in scg_samples:
            x_counts = raw_packet_int16_to_adc_counts(x)
            y_counts = raw_packet_int16_to_adc_counts(y)
            z_counts = raw_packet_int16_to_adc_counts(z)

            x_g = adc_counts_to_g(x_counts)
            y_g = adc_counts_to_g(y_counts)
            z_g = adc_counts_to_g(z_counts)

            if self._filter_enabled:
                x_g, y_g, z_g = self._apply_filter_sample(x_g, y_g, z_g)

            self._scg_ts.append(int(ts))
            self._scg_x.append(x_g)
            self._scg_y.append(y_g)
            self._scg_z.append(z_g)
            self._sample_count += 1
            self._rate_count   += 1

            if self._is_recording and self._csv_writer is not None:
                self._csv_writer.writerow([
                    int(ts),
                    self._scg_x[-1],
                    self._scg_y[-1],
                    self._scg_z[-1],
                    0,
                ])
                self._record_samples += 1
                if self._record_first_ts is None:
                    self._record_first_ts = int(ts)
                self._record_last_ts = int(ts)

        for ts in beat_timestamps:
            self._beat_ts.append(ts)
            if self._last_beat_ts is not None:
                interval_ms = ts - self._last_beat_ts
                if 300 < interval_ms < 2000:   # sanity: 30–200 BPM
                    self._beat_intervals.append(interval_ms)
            self._last_beat_ts = ts

            if self._is_recording and self._csv_writer is not None:
                self._csv_writer.writerow([int(ts), "", "", "", 1])

        self._plot_dirty = True
        self._refresh_stats()

    def _on_plot_timer(self):
        if self._is_recording:
            return
        if not self._plot_dirty:
            return
        self._refresh_plots()
        self._refresh_segment_plot()
        self._plot_dirty = False

    def _start_save(self):
        if self._is_recording:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SCG Data",
            "scg_capture.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            self._csv_file = open(path, "w", newline="", encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Save Error", f"Could not open file:\n{e}")
            return

        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestamp_ms", "x_g", "y_g", "z_g", "beat_event"])

        self._record_path = path
        self._record_samples = 0
        self._record_first_ts = None
        self._record_last_ts = None
        self._is_recording = True

        self._record_elapsed_secs = 0
        self._record_timer_lbl.setText("REC TIME: 00:00")
        self._record_timer.start(1000)

        self._save_start_btn.setEnabled(False)
        self._save_stop_btn.setEnabled(True)
        self._status_lbl.setText("RECORDING")
        self._status_lbl.setStyleSheet(f"color:{ACCENT};")

    def _stop_save(self):
        if not self._is_recording:
            return

        self._is_recording = False
        self._record_timer.stop()

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()

        self._csv_file = None
        self._csv_writer = None

        expected_hz = float(SAMPLE_RATE)
        actual_hz = 0.0
        if (
            self._record_samples >= 2
            and self._record_first_ts is not None
            and self._record_last_ts is not None
        ):
            duration_s = (self._record_last_ts - self._record_first_ts) / 1000.0
            if duration_s > 0:
                actual_hz = (self._record_samples - 1) / duration_s

        diff_hz = actual_hz - expected_hz
        diff_pct = (diff_hz / expected_hz * 100.0) if expected_hz > 0 else 0.0

        self._save_start_btn.setEnabled(True)
        self._save_stop_btn.setEnabled(False)
        elapsed_s = getattr(self, '_record_elapsed_secs', 0)
        self._record_timer_lbl.setText(f"REC TIME: {elapsed_s // 60:02d}:{elapsed_s % 60:02d}")
        self._status_lbl.setText("CONNECTED")
        self._status_lbl.setStyleSheet(f"color:{GREEN};")

        self._plot_dirty = True

        QMessageBox.information(
            self,
            "Capture Summary",
            "CSV saved to:\n"
            f"{self._record_path}\n\n"
            f"Expected rate: {expected_hz:.2f} Hz\n"
            f"Actual received rate: {actual_hz:.2f} Hz\n"
            f"Difference: {diff_hz:+.2f} Hz ({diff_pct:+.2f}%)\n"
            f"SCG samples saved: {self._record_samples}\n"
            f"Elapsed time: {elapsed_s // 60:02d}:{elapsed_s % 60:02d}"
        )

    def _update_record_timer(self):
        if not self._is_recording:
            return
        self._record_elapsed_secs = getattr(self, '_record_elapsed_secs', 0) + 1
        mm = self._record_elapsed_secs // 60
        ss = self._record_elapsed_secs % 60
        self._record_timer_lbl.setText(f"REC TIME: {mm:02d}:{ss:02d}")

    def _decimate(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(y) <= MAX_PLOT_POINTS:
            return x, y
        step = max(1, len(y) // MAX_PLOT_POINTS)
        return x[::step], y[::step]

    def _refresh_plots(self):
        valid_n = min(self._sample_count, WINDOW_N)
        if valid_n < 2:
            return

        ts_valid = np.array(list(self._scg_ts)[-valid_n:], dtype=np.int64)
        t_axis = (ts_valid - ts_valid[-1]).astype(np.float32) / 1000.0

        arrays = [
            np.array(list(self._scg_x)[-valid_n:], dtype=np.float32),
            np.array(list(self._scg_y)[-valid_n:], dtype=np.float32),
            np.array(list(self._scg_z)[-valid_n:], dtype=np.float32),
        ]

        for i, (curve, arr) in enumerate(zip(self._curves, arrays)):
            x_view, y_view = self._decimate(t_axis, arr)
            curve.setData(x_view, y_view)

            # Beat markers: remove old, add new within visible window
            pw = self._plots[i]
            for line in self._beat_lines[i]:
                pw.removeItem(line)
            self._beat_lines[i].clear()

            if self._last_beat_ts is not None and self._sample_count > 0:
                # Anchor beat ages to the live SCG clock so beat markers scroll continuously.
                now_ts = self._scg_ts[-1]
                for b_ts in self._beat_ts[-20:]:   # only last 20 beats
                    age_s = (now_ts - b_ts) / 1000.0
                    if 0 <= age_s <= WINDOW_SECS:
                        t_pos = -age_s
                        line = pg.InfiniteLine(
                            pos=t_pos,
                            angle=90,
                            pen=pg.mkPen(ACCENT2, width=1, style=Qt.DashLine)
                        )
                        pw.addItem(line)
                        self._beat_lines[i].append(line)

    def _refresh_segment_plot(self):
        if len(self._beat_ts) < 2:
            for curve in self._segment_curves:
                curve.setData([], [])
            return

        start_ts = self._beat_ts[-2]
        end_ts = self._beat_ts[-1]
        if end_ts <= start_ts:
            return

        ts = np.array(self._scg_ts, dtype=np.int64)
        mask = (ts >= start_ts) & (ts <= end_ts)
        if np.count_nonzero(mask) < 2:
            for curve in self._segment_curves:
                curve.setData([], [])
            return

        seg_t = (ts[mask] - start_ts).astype(np.float32) / 1000.0
        seg_x = np.array(self._scg_x, dtype=np.float32)[mask]
        seg_y = np.array(self._scg_y, dtype=np.float32)[mask]
        seg_z = np.array(self._scg_z, dtype=np.float32)[mask]

        seg_t_view, seg_x_view = self._decimate(seg_t, seg_x)
        _, seg_y_view = self._decimate(seg_t, seg_y)
        _, seg_z_view = self._decimate(seg_t, seg_z)

        self._segment_curves[0].setData(seg_t_view, seg_x_view)
        self._segment_curves[1].setData(seg_t_view, seg_y_view)
        self._segment_curves[2].setData(seg_t_view, seg_z_view)
        self._segment_plot.setXRange(0, float(seg_t_view[-1]), padding=0.02)

    def _refresh_stats(self):
        self._stat_beats.setText(str(len(self._beat_ts)))

        if len(self._beat_intervals) >= 2:
            avg_interval = np.mean(self._beat_intervals)
            bpm = 60000.0 / avg_interval
            self._bpm_label.setText(f"{bpm:.0f}")
        else:
            self._bpm_label.setText("--")

    def _update_rate(self):
        self._stat_rate.setText(f"{self._rate_count} Hz")
        self._rate_count = 0

    def _clear_data(self):
        self._scg_x = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
        self._scg_y = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
        self._scg_z = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
        self._scg_ts = deque([0] * WINDOW_N, maxlen=WINDOW_N)
        self._reset_filter_state()
        self._beat_ts.clear()
        self._beat_intervals.clear()
        self._last_beat_ts = None
        self._sample_count = 0
        self._bpm_label.setText("--")
        self._stat_beats.setText("0")
        for curve in self._segment_curves:
            curve.setData([], [])

    def closeEvent(self, event):
        self._on_disconnect()
        event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pg.setConfigOptions(antialias=True, useOpenGL=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette base so non-styled widgets don't flash white
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.Base,            QColor(BG_CARD))
    palette.setColor(QPalette.AlternateBase,   QColor(BG_PANEL))
    palette.setColor(QPalette.Text,            QColor(TEXT))
    palette.setColor(QPalette.Button,          QColor(BG_CARD))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor(BG))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())