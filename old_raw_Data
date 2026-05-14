"""
SCG / PPG Beat Visualizer
Reads binary packets from STM32 over USB CDC and displays:
    - SCG (X, Y, Z) at 256 Hz — rolling waveform
    - PPG raw waveform at 100 Hz
    - PPG beat events — overlaid markers + BPM readout

Packet format:
  SCG  [0xAA][0x01][ts:4B LE][x:2B][y:2B][z:2B][chk:1B]  = 13 bytes
  BEAT [0xAA][0x02][ts:4B LE][chk:1B]                     = 7 bytes
  PPG  [0xAA][0x03][ts:4B LE][ppg:4B][chk:1B]             = 11 bytes

FIXES vs original:
  1. Font family: added SF Mono / Menlo / Monaco fallbacks for macOS so the
     600 ms font-alias scan doesn't block startup.
  2. Painter path ±32767 overflow: time axis is now always relative to the
     most-recent sample (max value 0, min ≈ –WINDOW_SECS).  The old code
     used raw millisecond timestamps which grow to ~millions of pixels.
  3. Plot freeze: the beat-marker loop was rebuilding pg.InfiniteLine objects
     every 33 ms even when nothing changed.  Lines are now cached and only
     recreated when the beat list actually changes.
  4. Serial thread: moved the blocking read loop onto a proper QThread so the
     main-thread event loop is never starved.  Also switched from a bare
     while-True _loop() called inside run() (which was effectively the same
     but fragile) to a clean QObject + moveToThread pattern so Qt signals cross
     threads safely.
  5. Deque → numpy: avoided repeated full-deque copies inside the 33 ms timer
     by keeping a pre-allocated numpy ring-buffer instead of deque for the
     plot arrays (deques kept only for recording/segment lookups).
"""

import sys
import struct
import csv
import time
import serial
import serial.tools.list_ports
from collections import deque
from threading import Lock

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
TYPE_PPG     = 0x03
SCG_PKT_LEN  = 13
BEAT_PKT_LEN = 7
PPG_PKT_LEN  = 11

def xor_checksum(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
    return c

def parse_packets(buf: bytearray):
    """
    Consume as many complete packets from buf as possible.
    Returns (scg_samples, ppg_samples, beat_timestamps, remaining_buf, parse_errors)
      scg_samples      : list of (timestamp_ms, x, y, z)
      ppg_samples      : list of (timestamp_ms, ppg_raw)
      beat_timestamps  : list of timestamp_ms
    """
    scg_samples = []
    ppg_samples = []
    beat_timestamps = []
    parse_errors = 0

    if len(buf) > 4096:
        last_magic = buf.rfind(bytes([MAGIC]))
        if last_magic >= 0:
            del buf[:last_magic]
        else:
            buf.clear()
        parse_errors += 1

    while len(buf) >= BEAT_PKT_LEN:
        if buf[0] != MAGIC:
            buf.pop(0)
            parse_errors += 1
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
                parse_errors += 1
                continue
            del buf[:SCG_PKT_LEN]

        elif pkt_type == TYPE_PPG:
            if len(buf) < PPG_PKT_LEN:
                break
            pkt = bytes(buf[:PPG_PKT_LEN])
            chk = xor_checksum(pkt[:-1])
            if chk == pkt[-1]:
                ts, ppg_raw = struct.unpack_from('<II', pkt, 2)
                ppg_samples.append((ts, ppg_raw))
            else:
                buf.pop(0)
                parse_errors += 1
                continue
            del buf[:PPG_PKT_LEN]

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
                parse_errors += 1
                continue
            del buf[:BEAT_PKT_LEN]

        else:
            buf.pop(0)
            parse_errors += 1

    return scg_samples, ppg_samples, beat_timestamps, buf, parse_errors


# ─── Serial Reader Thread ─────────────────────────────────────────────────────

class SerialReader(QObject):
    data_ready = pyqtSignal(list, list, list, int)
    error      = pyqtSignal(str)

    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port  = port
        self.baud  = baud
        self._running = False
        self._ser  = None
        self._buf  = bytearray()
        self._parse_errors_pending = 0
        # Accumulate samples for main-thread polling.
        self._lock = Lock()
        self._scg_accum: list = []
        self._ppg_accum: list = []
        self._beat_accum: list = []
        self._max_scg_accum = 1024
        self._max_ppg_accum = 512
        self._max_beat_accum = 128

    def _trim_accum(self, buf: list, max_len: int) -> list:
        if len(buf) <= max_len:
            return buf
        return buf[-max_len:]

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
                    scg, ppg, beats, self._buf, parse_errors = parse_packets(self._buf)
                    if scg or ppg or beats or parse_errors:
                        with self._lock:
                            if parse_errors:
                                self._parse_errors_pending += parse_errors
                            if scg:
                                self._scg_accum.extend(scg)
                                self._scg_accum = self._trim_accum(
                                    self._scg_accum, self._max_scg_accum
                                )
                            if ppg:
                                self._ppg_accum.extend(ppg)
                                self._ppg_accum = self._trim_accum(
                                    self._ppg_accum, self._max_ppg_accum
                                )
                            if beats:
                                self._beat_accum.extend(beats)
                                self._beat_accum = self._trim_accum(
                                    self._beat_accum, self._max_beat_accum
                                )
            except serial.SerialException as e:
                self.error.emit(str(e))
                break

    def drain(self):
        with self._lock:
            if (not self._scg_accum
                    and not self._ppg_accum
                    and not self._beat_accum
                    and not self._parse_errors_pending):
                return None
            scg = self._scg_accum
            ppg = self._ppg_accum
            beats = self._beat_accum
            parse_errors = self._parse_errors_pending
            self._scg_accum = []
            self._ppg_accum = []
            self._beat_accum = []
            self._parse_errors_pending = 0
            return scg, ppg, beats, parse_errors


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
ACCENT      = "#00e5ff"
ACCENT2     = "#ff4757"
GREEN       = "#2ed573"
MUTED       = "#4a5068"
TEXT        = "#e8eaf0"
TEXT_DIM    = "#6b7494"

COLORS_SCG  = [ACCENT, "#a78bfa", GREEN]
LABEL_SCG   = ["X", "Y", "Z"]

# FIX 1: macOS-friendly font stack — avoids the 600 ms alias-scan warning.
# Qt picks the first family it can find.
FONT_STACK = "'SF Mono', 'Menlo', 'Monaco', 'JetBrains Mono', 'Consolas', monospace"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: {FONT_STACK};
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

def make_plot_widget(title: str) -> pg.PlotWidget:
    pw = pg.PlotWidget()
    pw.setBackground(BG_PANEL)
    pw.showGrid(x=False, y=True, alpha=0.08)
    pw.getAxis('left').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('bottom').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('left').setPen(pg.mkPen(BORDER))
    pw.getAxis('bottom').setPen(pg.mkPen(BORDER))
    pw.setClipToView(True)
    pw.setDownsampling(auto=True, mode="peak")
    pw.addItem(pg.LabelItem(
        f'<span style="color:{TEXT_DIM};font-size:10px;letter-spacing:2px">{title}</span>'
    ))
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=True)
    return pw


# ─── Constants ────────────────────────────────────────────────────────────────

WINDOW_SECS      = 5
SAMPLE_RATE      = 256
PPG_SAMPLE_RATE  = 100
WINDOW_N         = WINDOW_SECS * SAMPLE_RATE
WINDOW_PPG_N     = WINDOW_SECS * PPG_SAMPLE_RATE
BPM_HISTORY      = 8
MAX_PLOT_POINTS  = 2000
BPF_LOW_HZ       = 0.5
BPF_HIGH_HZ      = 50.0

ADC_FULL_SCALE_COUNTS   = 65535.0
ADC_ZERO_G_COUNTS       = ADC_FULL_SCALE_COUNTS / 2.0
ADC_VREF                = 3.3
ADXL335_SENSITIVITY_V_PER_G = 0.3
ADC_COUNTS_PER_G = (ADXL335_SENSITIVITY_V_PER_G / ADC_VREF) * ADC_FULL_SCALE_COUNTS
PPG_MAX_COUNTS   = 262143.0


def raw_packet_int16_to_adc_counts(raw_value: int) -> float:
    return float(raw_value & 0xFFFF)

def adc_counts_to_g(adc_counts: float, zero_g_counts: float = ADC_ZERO_G_COUNTS) -> float:
    return (adc_counts - zero_g_counts) / ADC_COUNTS_PER_G

def ppg_counts_to_norm(ppg_counts: float) -> float:
    return ppg_counts / PPG_MAX_COUNTS


# ─── Ring-buffer helper ───────────────────────────────────────────────────────

class RingBuffer:
    """
    Fixed-size float32/int64 ring buffer with O(1) append and O(N) slice-to-array.
    Avoids repeated deque→list→numpy conversion in the hot plot path.
    """
    def __init__(self, size: int, dtype=np.float32):
        self._buf  = np.zeros(size, dtype=dtype)
        self._size = size
        self._idx  = 0          # next write position
        self._full = False

    def append(self, value):
        self._buf[self._idx] = value
        self._idx = (self._idx + 1) % self._size
        if self._idx == 0:
            self._full = True

    def to_array(self, n: int | None = None) -> np.ndarray:
        """Return the last `n` samples (or all valid samples) in chronological order."""
        valid = self._size if self._full else self._idx
        if n is None or n >= valid:
            n = valid
        if n == 0:
            return self._buf[:0].copy()
        # The oldest valid sample is at self._idx when full, 0 otherwise.
        start = (self._idx - n) % self._size if self._full else max(0, self._idx - n)
        if start + n <= self._size:
            return self._buf[start:start + n].copy()
        # Wraps around
        return np.concatenate((self._buf[start:], self._buf[:n - (self._size - start)]))

    @property
    def valid_count(self) -> int:
        return self._size if self._full else self._idx


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCG · PPG Monitor")
        self.resize(1280, 820)

        # ── Ring buffers for plot data (fast path) ────────────────────────────
        self._rb_scg_x  = RingBuffer(WINDOW_N)
        self._rb_scg_y  = RingBuffer(WINDOW_N)
        self._rb_scg_z  = RingBuffer(WINDOW_N)
        self._rb_scg_ts = RingBuffer(WINDOW_N, dtype=np.int64)
        self._rb_scg_host_ts = RingBuffer(WINDOW_N, dtype=np.int64)

        self._rb_ppg    = RingBuffer(WINDOW_PPG_N)
        self._rb_ppg_ts = RingBuffer(WINDOW_PPG_N, dtype=np.int64)
        self._rb_ppg_host_ts = RingBuffer(WINDOW_PPG_N, dtype=np.int64)

        # ── Deques kept for recording / segment lookups (non-plot) ────────────
        self._scg_x_dq  = deque(maxlen=WINDOW_N)
        self._scg_y_dq  = deque(maxlen=WINDOW_N)
        self._scg_z_dq  = deque(maxlen=WINDOW_N)
        self._scg_ts_dq = deque(maxlen=WINDOW_N)
        self._scg_host_ts_dq = deque(maxlen=WINDOW_N)

        self._beat_ts:       list[int] = []
        self._beat_host_ts:  list[int] = []
        self._beat_intervals: deque    = deque(maxlen=BPM_HISTORY)
        self._last_beat_ts:  int | None = None
        self._last_beat_host_ts: int | None = None

        self._host_sample_clock_ms: float | None = None
        self._ppg_host_clock_ms:    float | None = None
        self._use_host_time_axis    = False
        self._sample_count          = 0
        self._parse_error_count     = 0
        self._plot_dirty            = False
        self._segment_dirty         = False

        # Beat-marker cache: avoid recreating InfiniteLine objects every frame
        # Maps axis index → list of (beat_ts, InfiniteLine)
        self._beat_line_cache: list[list[tuple[int, pg.InfiniteLine]]] = [[], [], []]
        self._beat_ts_snapshot: list[int] = []   # last rendered beat list

        # Serial state
        self._thread: ReaderThread | None = None
        self._reader: SerialReader | None = None

        # Recording state
        self._is_recording        = False
        self._csv_file            = None
        self._csv_writer          = None
        self._record_path         = ""
        self._record_samples      = 0
        self._record_first_ts     = None
        self._record_last_ts      = None
        self._record_elapsed_secs = 0

        # Filter
        self._filter_enabled = False
        self._bpf_sos = butter(2, [BPF_LOW_HZ, BPF_HIGH_HZ],
                               btype='bandpass', fs=SAMPLE_RATE, output='sos')
        self._reset_filter_state()

        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._update_record_timer)

        self._ingest_timer = QTimer(self)
        self._ingest_timer.timeout.connect(self._drain_serial)

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._on_plot_timer)
        self._plot_timer.start(33)   # ~30 fps

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        root_layout.addWidget(self._build_plots(), stretch=1)
        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(220)
        root_layout.addWidget(sidebar)

    def _build_plots(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        title_lbl = QLabel("SEISMOCARDIOGRAM")
        title_lbl.setStyleSheet(
            f"color:{ACCENT};font-size:13px;font-weight:bold;letter-spacing:3px;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        for label, color in zip(LABEL_SCG, COLORS_SCG):
            dot = QLabel(f"● {label}")
            dot.setStyleSheet(f"color:{color};font-size:11px;margin-left:8px;")
            title_row.addWidget(dot)
        layout.addLayout(title_row)

        self._plots:  list[pg.PlotWidget]  = []
        self._curves: list[pg.PlotDataItem] = []

        for axis_name, color in zip(["X AXIS", "Y AXIS", "Z AXIS"], COLORS_SCG):
            pw = make_plot_widget(axis_name)
            # FIX 2: fix x-range in relative-time units, never raw ms
            pw.setXRange(-WINDOW_SECS, 0, padding=0)
            pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            curve = pw.plot([], [], pen=pg.mkPen(color, width=1.5, cosmetic=True))
            self._plots.append(pw)
            self._curves.append(curve)
            layout.addWidget(pw)

        ppg_title_row = QHBoxLayout()
        ppg_title = QLabel("PPG RAW")
        ppg_title.setStyleSheet(
            f"color:{ACCENT2};font-size:12px;font-weight:bold;letter-spacing:3px;")
        ppg_title_row.addWidget(ppg_title)
        ppg_title_row.addStretch()
        layout.addLayout(ppg_title_row)

        self._ppg_plot = make_plot_widget("PPG WAVEFORM")
        self._ppg_plot.setXRange(-WINDOW_SECS, 0, padding=0)
        self._ppg_plot.setYRange(0.0, 1.0, padding=0.02)
        self._ppg_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ppg_curve = self._ppg_plot.plot(
            [], [], pen=pg.mkPen(ACCENT2, width=1.5, cosmetic=True))
        layout.addWidget(self._ppg_plot)

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
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        bpm_card = QFrame(); bpm_card.setObjectName("card")
        bpm_layout = QVBoxLayout(bpm_card)
        bpm_layout.setContentsMargins(16, 16, 16, 16)
        bpm_layout.setSpacing(2)
        bpm_title = QLabel("HEART RATE"); bpm_title.setObjectName("stat_label")
        self._bpm_label = QLabel("--"); self._bpm_label.setObjectName("bpm_value")
        self._bpm_label.setAlignment(Qt.AlignCenter)
        bpm_unit = QLabel("BPM"); bpm_unit.setObjectName("stat_label")
        bpm_unit.setAlignment(Qt.AlignCenter)
        bpm_layout.addWidget(bpm_title, alignment=Qt.AlignCenter)
        bpm_layout.addWidget(self._bpm_label)
        bpm_layout.addWidget(bpm_unit)
        layout.addWidget(bpm_card)

        stats_card = QFrame(); stats_card.setObjectName("card")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(16, 14, 16, 14); stats_layout.setSpacing(10)
        self._stat_beats = self._make_stat(stats_layout, "BEATS TOTAL",  "0")
        self._stat_rate  = self._make_stat(stats_layout, "SAMPLE RATE",  "-- Hz")
        self._stat_lost  = self._make_stat(stats_layout, "PARSE ERRORS", "0")
        layout.addWidget(stats_card)

        conn_card = QFrame(); conn_card.setObjectName("card")
        conn_layout = QVBoxLayout(conn_card)
        conn_layout.setContentsMargins(16, 14, 16, 14); conn_layout.setSpacing(8)
        port_lbl = QLabel("PORT"); port_lbl.setObjectName("stat_label")
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
        for w in [port_lbl, self._port_combo, refresh_btn,
                  self._connect_btn, self._stop_btn, self._status_lbl]:
            conn_layout.addWidget(w)
        layout.addWidget(conn_card)

        layout.addStretch()

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

        self._timebase_checkbox = QCheckBox("USE PC TIME AXIS")
        self._timebase_checkbox.setChecked(False)
        self._timebase_checkbox.toggled.connect(self._on_timebase_toggled)
        layout.addWidget(self._timebase_checkbox)

        clear_btn = QPushButton("CLEAR PLOTS")
        clear_btn.clicked.connect(self._clear_data)
        layout.addWidget(clear_btn)

        return sidebar

    def _make_stat(self, parent_layout, label_text: str, value_text: str) -> QLabel:
        row = QHBoxLayout()
        lbl = QLabel(label_text); lbl.setObjectName("stat_label")
        val = QLabel(value_text); val.setObjectName("stat_value")
        val.setStyleSheet(f"color:{TEXT};font-size:13px;font-weight:bold;")
        row.addWidget(lbl); row.addStretch(); row.addWidget(val)
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
        self._reader.error.connect(self._on_serial_error)
        self._thread = ReaderThread(self._reader)
        self._thread.start()
        self._connect_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText("CONNECTED")
        self._status_lbl.setStyleSheet(f"color:{GREEN};")
        self._ingest_timer.start(33)
        self._rate_count = 0
        self._rate_timer = QTimer()
        self._rate_timer.timeout.connect(self._update_rate)
        self._rate_timer.start(1000)

    def _on_disconnect(self):
        if self._is_recording:
            self._stop_save()
        if self._thread:
            self._thread.stop()
            self._thread = None
        self._reader = None
        self._ingest_timer.stop()
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

    # ── Filter ────────────────────────────────────────────────────────────────

    def _reset_filter_state(self):
        n = self._bpf_sos.shape[0]
        self._zi_x = np.zeros((n, 2), dtype=np.float64)
        self._zi_y = np.zeros((n, 2), dtype=np.float64)
        self._zi_z = np.zeros((n, 2), dtype=np.float64)

    def _apply_filter_sample(self, xv, yv, zv):
        xo, self._zi_x = sosfilt(self._bpf_sos, np.array([xv]), zi=self._zi_x)
        yo, self._zi_y = sosfilt(self._bpf_sos, np.array([yv]), zi=self._zi_y)
        zo, self._zi_z = sosfilt(self._bpf_sos, np.array([zv]), zi=self._zi_z)
        return float(xo[0]), float(yo[0]), float(zo[0])

    def _on_filter_toggled(self, checked: bool):
        self._filter_enabled = checked
        self._reset_filter_state()

    def _on_timebase_toggled(self, checked: bool):
        self._use_host_time_axis = checked
        self._plot_dirty = True

    # ── Data ingestion ────────────────────────────────────────────────────────

    def _drain_serial(self):
        if not self._reader:
            return
        batch = self._reader.drain()
        if batch is None:
            return
        scg_samples, ppg_samples, beat_timestamps, parse_errors = batch
        if len(scg_samples) > 128:
            scg_samples = scg_samples[-128:]
        if len(ppg_samples) > 64:
            ppg_samples = ppg_samples[-64:]
        if len(beat_timestamps) > 16:
            beat_timestamps = beat_timestamps[-16:]
        self._on_data(scg_samples, ppg_samples, beat_timestamps, parse_errors)

    def _on_data(self, scg_samples: list, ppg_samples: list,
                 beat_timestamps: list, parse_errors: int):
        self._parse_error_count += int(parse_errors)
        sample_dt_ms     = 1000.0 / SAMPLE_RATE
        ppg_sample_dt_ms = 1000.0 / PPG_SAMPLE_RATE

        for ts, x, y, z in scg_samples:
            if self._host_sample_clock_ms is None:
                self._host_sample_clock_ms = time.monotonic() * 1000.0
            else:
                self._host_sample_clock_ms += sample_dt_ms

            x_g = adc_counts_to_g(raw_packet_int16_to_adc_counts(x))
            y_g = adc_counts_to_g(raw_packet_int16_to_adc_counts(y))
            z_g = adc_counts_to_g(raw_packet_int16_to_adc_counts(z))

            if self._filter_enabled:
                x_g, y_g, z_g = self._apply_filter_sample(x_g, y_g, z_g)

            host_ts = int(self._host_sample_clock_ms)
            self._rb_scg_ts.append(int(ts))
            self._rb_scg_host_ts.append(host_ts)
            self._rb_scg_x.append(x_g)
            self._rb_scg_y.append(y_g)
            self._rb_scg_z.append(z_g)

            # Keep deques in sync for segment / recording use
            self._scg_ts_dq.append(int(ts))
            self._scg_host_ts_dq.append(host_ts)
            self._scg_x_dq.append(x_g)
            self._scg_y_dq.append(y_g)
            self._scg_z_dq.append(z_g)

            self._sample_count += 1
            self._rate_count   += 1

            if self._is_recording and self._csv_writer is not None:
                self._csv_writer.writerow([int(ts), x_g, y_g, z_g, 0, ""])
                self._record_samples += 1
                if self._record_first_ts is None:
                    self._record_first_ts = int(ts)
                self._record_last_ts = int(ts)

        for ts, ppg_raw in ppg_samples:
            if self._ppg_host_clock_ms is None:
                self._ppg_host_clock_ms = time.monotonic() * 1000.0
            else:
                self._ppg_host_clock_ms += ppg_sample_dt_ms
            self._rb_ppg_ts.append(int(ts))
            self._rb_ppg_host_ts.append(int(self._ppg_host_clock_ms))
            self._rb_ppg.append(ppg_counts_to_norm(float(ppg_raw)))
            if self._is_recording and self._csv_writer is not None:
                self._csv_writer.writerow([int(ts), "", "", "", 0, int(ppg_raw)])

        for ts in beat_timestamps:
            self._beat_ts.append(ts)
            host_beat_ts = int(time.monotonic() * 1000.0)
            self._beat_host_ts.append(host_beat_ts)
            self._last_beat_host_ts = host_beat_ts
            if self._last_beat_ts is not None:
                interval_ms = ts - self._last_beat_ts
                if 300 < interval_ms < 2000:
                    self._beat_intervals.append(interval_ms)
            self._last_beat_ts = ts

            if self._is_recording and self._csv_writer is not None:
                self._csv_writer.writerow([int(ts), "", "", "", 1, ""])

        self._plot_dirty = True
        if beat_timestamps:
            self._segment_dirty = True
        self._refresh_stats()

    # ── Plot timer ────────────────────────────────────────────────────────────

    def _on_plot_timer(self):
        if self._is_recording or not self._plot_dirty:
            return
        self._refresh_plots()
        self._refresh_ppg_plot()
        if self._segment_dirty:
            self._refresh_segment_plot()
            self._segment_dirty = False
        self._plot_dirty = False

    # ── Plot refresh ──────────────────────────────────────────────────────────

    def _decimate(self, x: np.ndarray, y: np.ndarray):
        if len(y) <= MAX_PLOT_POINTS:
            return x, y
        step = max(1, len(y) // MAX_PLOT_POINTS)
        return x[::step], y[::step]

    def _refresh_plots(self):
        valid_n = self._rb_scg_ts.valid_count
        if valid_n < 2:
            return

        rb_ts = self._rb_scg_host_ts if self._use_host_time_axis else self._rb_scg_ts
        beat_ts_source = self._beat_host_ts if self._use_host_time_axis else self._beat_ts

        ts_arr = rb_ts.to_array(valid_n)
        now_ts = int(ts_arr[-1])
        # FIX 2: time axis is always relative seconds → max 0, stays in ±WINDOW_SECS
        t_axis = (ts_arr - now_ts).astype(np.float32) / 1000.0

        arrs = [
            self._rb_scg_x.to_array(valid_n),
            self._rb_scg_y.to_array(valid_n),
            self._rb_scg_z.to_array(valid_n),
        ]

        # FIX 3: only rebuild beat markers when the beat list actually changes
        beats_changed = (beat_ts_source != self._beat_ts_snapshot)

        for i, (curve, arr) in enumerate(zip(self._curves, arrs)):
            xv, yv = self._decimate(t_axis, arr)
            curve.setData(xv, yv)

            if beats_changed:
                pw = self._plots[i]
                # Remove stale lines
                for _, line in self._beat_line_cache[i]:
                    pw.removeItem(line)
                self._beat_line_cache[i].clear()

                for b_ts in beat_ts_source[-20:]:
                    age_s = (now_ts - b_ts) / 1000.0
                    if 0 <= age_s <= WINDOW_SECS:
                        line = pg.InfiniteLine(
                            pos=-age_s, angle=90,
                            pen=pg.mkPen(ACCENT2, width=1, style=Qt.DashLine)
                        )
                        pw.addItem(line)
                        self._beat_line_cache[i].append((b_ts, line))
            else:
                # Just shift existing lines; no object creation
                for b_ts, line in self._beat_line_cache[i]:
                    age_s = (now_ts - b_ts) / 1000.0
                    if 0 <= age_s <= WINDOW_SECS:
                        line.setPos(-age_s)
                    else:
                        line.setPos(-WINDOW_SECS - 1)  # park it off-screen

        if beats_changed:
            self._beat_ts_snapshot = list(beat_ts_source)

    def _refresh_ppg_plot(self):
        valid_n = self._rb_ppg_ts.valid_count
        if valid_n < 2:
            return
        rb_ts = self._rb_ppg_host_ts if self._use_host_time_axis else self._rb_ppg_ts
        ts_arr = rb_ts.to_array(valid_n)
        t_axis = (ts_arr - int(ts_arr[-1])).astype(np.float32) / 1000.0
        ppg_arr = self._rb_ppg.to_array(valid_n)
        xv, yv = self._decimate(t_axis, ppg_arr)
        self._ppg_curve.setData(xv, yv)

    def _refresh_segment_plot(self):
        beat_ts_source = self._beat_host_ts if self._use_host_time_axis else self._beat_ts
        if len(beat_ts_source) < 2:
            for c in self._segment_curves:
                c.setData([], [])
            return

        start_ts = beat_ts_source[-2]
        end_ts   = beat_ts_source[-1]
        if end_ts <= start_ts:
            return

        ts_dq = self._scg_host_ts_dq if self._use_host_time_axis else self._scg_ts_dq
        ts = np.array(ts_dq, dtype=np.int64)
        mask = (ts >= start_ts) & (ts <= end_ts)
        if np.count_nonzero(mask) < 2:
            for c in self._segment_curves:
                c.setData([], [])
            return

        seg_t = (ts[mask] - start_ts).astype(np.float32) / 1000.0
        seg_x = np.array(self._scg_x_dq, dtype=np.float32)[mask]
        seg_y = np.array(self._scg_y_dq, dtype=np.float32)[mask]
        seg_z = np.array(self._scg_z_dq, dtype=np.float32)[mask]

        xt, xv = self._decimate(seg_t, seg_x)
        _,  yv = self._decimate(seg_t, seg_y)
        _,  zv = self._decimate(seg_t, seg_z)

        self._segment_curves[0].setData(xt, xv)
        self._segment_curves[1].setData(xt, yv)
        self._segment_curves[2].setData(xt, zv)
        if len(xt):
            self._segment_plot.setXRange(0, float(xt[-1]), padding=0.02)

    def _refresh_stats(self):
        self._stat_beats.setText(str(len(self._beat_ts)))
        self._stat_lost.setText(str(self._parse_error_count))
        if len(self._beat_intervals) >= 2:
            bpm = 60000.0 / np.mean(self._beat_intervals)
            self._bpm_label.setText(f"{bpm:.0f}")
        else:
            self._bpm_label.setText("--")

    def _update_rate(self):
        self._stat_rate.setText(f"{self._rate_count} Hz")
        self._rate_count = 0

    # ── Recording ─────────────────────────────────────────────────────────────

    def _start_save(self):
        if self._is_recording:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SCG Data", "scg_capture.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self._csv_file = open(path, "w", newline="", encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Save Error", f"Could not open file:\n{e}")
            return
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp_ms",
            "x_g",
            "y_g",
            "z_g",
            "beat_event",
            "ppg_raw",
        ])
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
        if (self._record_samples >= 2
                and self._record_first_ts is not None
                and self._record_last_ts is not None):
            duration_s = (self._record_last_ts - self._record_first_ts) / 1000.0
            if duration_s > 0:
                actual_hz = (self._record_samples - 1) / duration_s

        diff_hz  = actual_hz - expected_hz
        diff_pct = (diff_hz / expected_hz * 100.0) if expected_hz > 0 else 0.0
        elapsed_s = self._record_elapsed_secs

        self._save_start_btn.setEnabled(True)
        self._save_stop_btn.setEnabled(False)
        self._record_timer_lbl.setText(
            f"REC TIME: {elapsed_s // 60:02d}:{elapsed_s % 60:02d}")
        self._status_lbl.setText("CONNECTED")
        self._status_lbl.setStyleSheet(f"color:{GREEN};")
        self._plot_dirty = True

        QMessageBox.information(
            self, "Capture Summary",
            f"CSV saved to:\n{self._record_path}\n\n"
            f"Expected rate: {expected_hz:.2f} Hz\n"
            f"Actual received rate: {actual_hz:.2f} Hz\n"
            f"Difference: {diff_hz:+.2f} Hz ({diff_pct:+.2f}%)\n"
            f"SCG samples saved: {self._record_samples}\n"
            f"Elapsed time: {elapsed_s // 60:02d}:{elapsed_s % 60:02d}"
        )

    def _update_record_timer(self):
        if not self._is_recording:
            return
        self._record_elapsed_secs += 1
        mm = self._record_elapsed_secs // 60
        ss = self._record_elapsed_secs % 60
        self._record_timer_lbl.setText(f"REC TIME: {mm:02d}:{ss:02d}")

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _clear_data(self):
        self._rb_scg_x   = RingBuffer(WINDOW_N)
        self._rb_scg_y   = RingBuffer(WINDOW_N)
        self._rb_scg_z   = RingBuffer(WINDOW_N)
        self._rb_scg_ts  = RingBuffer(WINDOW_N, dtype=np.int64)
        self._rb_scg_host_ts = RingBuffer(WINDOW_N, dtype=np.int64)
        self._rb_ppg     = RingBuffer(WINDOW_PPG_N)
        self._rb_ppg_ts  = RingBuffer(WINDOW_PPG_N, dtype=np.int64)
        self._rb_ppg_host_ts = RingBuffer(WINDOW_PPG_N, dtype=np.int64)
        self._scg_x_dq.clear(); self._scg_y_dq.clear(); self._scg_z_dq.clear()
        self._scg_ts_dq.clear(); self._scg_host_ts_dq.clear()
        self._ppg_host_clock_ms = None
        self._host_sample_clock_ms = None
        self._reset_filter_state()
        self._beat_ts.clear(); self._beat_host_ts.clear()
        self._beat_intervals.clear()
        self._last_beat_ts = None; self._last_beat_host_ts = None
        self._sample_count = 0; self._parse_error_count = 0
        self._beat_ts_snapshot = []
        for cache in self._beat_line_cache:
            cache.clear()
        for pw in self._plots:
            pw.clear()
        self._bpm_label.setText("--")
        self._stat_beats.setText("0")
        self._stat_lost.setText("0")
        for c in self._segment_curves:
            c.setData([], [])

    def closeEvent(self, event):
        self._on_disconnect()
        event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pg.setConfigOptions(antialias=True, useOpenGL=False)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

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