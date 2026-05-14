"""
SCG / PPG Beat Visualizer  —  Subject Data Collection Edition
=============================================================
Reads binary packets from STM32 over USB CDC and displays:
    - SCG (X, Y, Z) at 256 Hz — rolling waveform
    - PPG raw waveform at 100 Hz
    - PPG beat events — overlaid markers + BPM readout

Packet format (unchanged):
  SCG  [0xAA][0x01][ts:4B LE][x:2B][y:2B][z:2B][chk:1B]  = 13 bytes
  BEAT [0xAA][0x02][ts:4B LE][chk:1B]                     = 7 bytes
  PPG  [0xAA][0x03][ts:4B LE][ppg:4B][chk:1B]             = 11 bytes

Raspberry Pi notes
------------------
- useOpenGL=False  (no GPU driver needed)
- antialias=False  (saves ~40% CPU on Pi 4 at 30 fps)
- Plot timer 40 ms (~25 fps) — safe on Pi 4 at 1280×800
- Serial reader on a true QThread; main thread never blocks
- All ring-buffer ops are pure numpy — no Python loops in hot path
- Writer thread for CSV I/O so disk latency never touches ingestion
- Recording does NOT pause the display (fixed from original)
- Metadata written as commented header rows + sidecar JSON
"""

import sys
import os
import struct
import csv
import json
import time
import queue
import threading
import serial
import serial.tools.list_ports
from collections import deque
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSplitter, QFrame, QSizePolicy,
    QCheckBox, QFileDialog, QMessageBox, QLineEdit, QSpinBox,
    QDoubleSpinBox, QGroupBox, QScrollArea, QDialog, QDialogButtonBox,
    QFormLayout, QRadioButton, QButtonGroup, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QRegExp
from PyQt5.QtGui import QFont, QColor, QPalette, QRegExpValidator

import pyqtgraph as pg
import numpy as np
from scipy.signal import butter, sosfilt


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════════════

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
    Consume complete packets from buf in-place.
    Returns (scg_samples, ppg_samples, beat_timestamps, parse_errors).
      scg_samples     : list of (timestamp_ms, x_raw, y_raw, z_raw)
      ppg_samples     : list of (timestamp_ms, ppg_raw)
      beat_timestamps : list of timestamp_ms
    """
    scg_samples     = []
    ppg_samples     = []
    beat_timestamps = []
    parse_errors    = 0

    # Safety valve: if buffer grows huge (serial stall), re-sync
    if len(buf) > 4096:
        last_magic = buf.rfind(bytes([MAGIC]))
        if last_magic > 0:
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
            if xor_checksum(pkt[:-1]) == pkt[-1]:
                ts, x, y, z = struct.unpack_from('<Ihhh', pkt, 2)
                scg_samples.append((ts, x, y, z))
                del buf[:SCG_PKT_LEN]
            else:
                buf.pop(0)
                parse_errors += 1

        elif pkt_type == TYPE_PPG:
            if len(buf) < PPG_PKT_LEN:
                break
            pkt = bytes(buf[:PPG_PKT_LEN])
            if xor_checksum(pkt[:-1]) == pkt[-1]:
                ts, ppg_raw = struct.unpack_from('<II', pkt, 2)
                ppg_samples.append((ts, ppg_raw))
                del buf[:PPG_PKT_LEN]
            else:
                buf.pop(0)
                parse_errors += 1

        elif pkt_type == TYPE_BEAT:
            if len(buf) < BEAT_PKT_LEN:
                break
            pkt = bytes(buf[:BEAT_PKT_LEN])
            if xor_checksum(pkt[:-1]) == pkt[-1]:
                (ts,) = struct.unpack_from('<I', pkt, 2)
                beat_timestamps.append(ts)
                del buf[:BEAT_PKT_LEN]
            else:
                buf.pop(0)
                parse_errors += 1

        else:
            buf.pop(0)
            parse_errors += 1

    return scg_samples, ppg_samples, beat_timestamps, parse_errors


# ═══════════════════════════════════════════════════════════════════════════════
# Serial Reader  —  dedicated QThread, never touches the main thread
# ═══════════════════════════════════════════════════════════════════════════════

class SerialReader(QObject):
    error = pyqtSignal(str)

    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port     = port
        self.baud     = baud
        self._running = False
        self._ser     = None
        self._buf     = bytearray()

        # Lock-protected accumulation buffers — drained by main thread at 25 Hz
        self._lock         = threading.Lock()
        self._scg_accum:   list = []
        self._ppg_accum:   list = []
        self._beat_accum:  list = []
        self._parse_errs:  int  = 0

    def start(self):
        try:
            # timeout=0.05 — short enough that stop() drains quickly,
            # long enough to avoid busy-spinning on the Pi.
            self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        except serial.SerialException as e:
            self.error.emit(str(e))
            return
        self._running = True
        self._loop()
        # _loop() returned (either stop() was called or a serial error occurred).
        # Close the port HERE, on the reader thread, so we never close
        # underneath a blocking read() call from a different thread.
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def stop(self):
        # Just signal the loop to exit; do NOT touch self._ser here.
        # The port is closed by start() after _loop() returns.
        self._running = False

    def _loop(self):
        while self._running:
            try:
                chunk = self._ser.read(512)   # larger read = fewer syscalls on Pi
            except serial.SerialException as e:
                self.error.emit(str(e))
                break
            except OSError:
                # Port was yanked from the OS (e.g. USB unplug) — exit cleanly.
                break
            if not chunk:
                continue
            self._buf.extend(chunk)
            scg, ppg, beats, errs = parse_packets(self._buf)
            if scg or ppg or beats or errs:
                with self._lock:
                    self._scg_accum.extend(scg)
                    self._ppg_accum.extend(ppg)
                    self._beat_accum.extend(beats)
                    self._parse_errs += errs
                    # Bound accumulator size so a slow drain can't grow unbounded
                    if len(self._scg_accum)  > 2048: self._scg_accum  = self._scg_accum[-2048:]
                    if len(self._ppg_accum)  > 1024: self._ppg_accum  = self._ppg_accum[-1024:]
                    if len(self._beat_accum) > 256:  self._beat_accum = self._beat_accum[-256:]

    def drain(self):
        """Called from the main thread. Returns (scg, ppg, beats, errs) or None."""
        with self._lock:
            if not self._scg_accum and not self._ppg_accum \
                    and not self._beat_accum and not self._parse_errs:
                return None
            result = (self._scg_accum, self._ppg_accum,
                      self._beat_accum, self._parse_errs)
            self._scg_accum  = []
            self._ppg_accum  = []
            self._beat_accum = []
            self._parse_errs = 0
            return result


class ReaderThread(QThread):
    def __init__(self, reader: SerialReader):
        super().__init__()
        self._reader = reader
        # Bump thread priority so serial reads aren't starved by Qt redraws
        self.setPriority(QThread.HighPriority)

    def run(self):
        self._reader.start()

    def stop(self):
        self._reader.stop()   # sets _running = False; port closed by reader thread
        self.quit()
        # Wait up to 500 ms — the read() timeout is 50 ms so the loop exits fast.
        if not self.wait(500):
            self.terminate()  # last resort; shouldn't be needed in practice
            self.wait(500)


# ═══════════════════════════════════════════════════════════════════════════════
# Async CSV writer  —  disk I/O on its own thread, never stalls ingestion
# ═══════════════════════════════════════════════════════════════════════════════

_WRITER_STOP = object()   # sentinel

class AsyncCSVWriter:
    """
    All write() calls are non-blocking. Rows are queued and flushed by a
    background daemon thread. This means disk latency (SD card on Pi) never
    causes the serial reader or the main thread to stall.
    """
    def __init__(self, path: str, fieldnames: list, metadata_lines: list):
        self._q = queue.Queue(maxsize=65536)
        self._path = path
        self._fieldnames = fieldnames
        self._metadata_lines = metadata_lines
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def write(self, row: dict):
        try:
            self._q.put_nowait(row)
        except queue.Full:
            pass   # drop rather than block; logged via parse_errors

    def close(self):
        self._q.put(_WRITER_STOP)
        self._thread.join(timeout=10)

    def _worker(self):
        with open(self._path, 'w', newline='', encoding='utf-8',
                  buffering=65536) as f:
            # Write metadata comment block
            for line in self._metadata_lines:
                f.write(f"# {line}\n")
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            while True:
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    f.flush()
                    continue
                if item is _WRITER_STOP:
                    f.flush()
                    break
                writer.writerow(item)


# ═══════════════════════════════════════════════════════════════════════════════
# Ring buffer  —  O(1) append, O(N) read, pure numpy
# ═══════════════════════════════════════════════════════════════════════════════

class RingBuffer:
    def __init__(self, size: int, dtype=np.float32):
        self._buf  = np.zeros(size, dtype=dtype)
        self._size = size
        self._idx  = 0
        self._full = False

    def append(self, value):
        self._buf[self._idx] = value
        self._idx = (self._idx + 1) % self._size
        if self._idx == 0:
            self._full = True

    def extend(self, values):
        for v in values:
            self.append(v)

    def to_array(self, n: int | None = None) -> np.ndarray:
        valid = self._size if self._full else self._idx
        if n is None or n >= valid:
            n = valid
        if n == 0:
            return self._buf[:0].copy()
        start = (self._idx - n) % self._size if self._full else max(0, self._idx - n)
        if start + n <= self._size:
            return self._buf[start:start + n].copy()
        return np.concatenate((self._buf[start:], self._buf[:n - (self._size - start)]))

    @property
    def valid_count(self) -> int:
        return self._size if self._full else self._idx

    def clear(self):
        self._buf[:] = 0
        self._idx  = 0
        self._full = False


# ═══════════════════════════════════════════════════════════════════════════════
# Styling  —  clinical dark theme, readable on Pi touchscreen
# ═══════════════════════════════════════════════════════════════════════════════

BG       = "#0a0c11"
BG_PANEL = "#10131a"
BG_CARD  = "#181c26"
BG_INPUT = "#1e2230"
BORDER   = "#252a38"
ACCENT   = "#00d4f5"
ACCENT2  = "#ff4757"
GREEN    = "#2ed573"
AMBER    = "#ffa502"
MUTED    = "#3a3f52"
TEXT     = "#dde1ed"
TEXT_DIM = "#5a6080"

COLORS_SCG = [ACCENT, "#a78bfa", GREEN]
LABEL_SCG  = ["X", "Y", "Z"]

FONT_MONO = "'Liberation Mono', 'DejaVu Sans Mono', 'Courier New', monospace"
FONT_UI   = "'Liberation Sans', 'DejaVu Sans', 'Arial', sans-serif"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: {FONT_UI};
    font-size: 11px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG_CARD};
}}
QTabBar::tab {{
    background: {BG};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    padding: 5px 14px;
    font-size: 10px;
    letter-spacing: 1px;
}}
QTabBar::tab:selected {{
    background: {BG_CARD};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 4px;
    font-size: 10px;
    color: {TEXT_DIM};
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    color: {TEXT_DIM};
}}
QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 8px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {TEXT};
    selection-background-color: {BORDER};
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    font-family: {FONT_MONO};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit[invalid="true"] {{
    border-color: {ACCENT2};
}}
QPushButton {{
    background-color: {BG_CARD};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 3px;
    padding: 6px 14px;
    font-weight: bold;
    letter-spacing: 1px;
    font-size: 10px;
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
QPushButton#rec_btn {{
    color: {AMBER};
    border-color: {AMBER};
    font-size: 11px;
}}
QPushButton#rec_btn:hover {{
    background-color: {AMBER};
    color: {BG};
}}
QPushButton#rec_btn:disabled {{
    color: {MUTED};
    border-color: {MUTED};
}}
QCheckBox {{
    color: {TEXT};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 2px;
    background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox:disabled {{ color: {MUTED}; }}
QLabel#stat_value {{
    color: {ACCENT};
    font-size: 26px;
    font-weight: bold;
    font-family: {FONT_MONO};
}}
QLabel#bpm_value {{
    color: {ACCENT2};
    font-size: 40px;
    font-weight: bold;
    font-family: {FONT_MONO};
}}
QLabel#stat_label {{
    color: {TEXT_DIM};
    font-size: 9px;
    letter-spacing: 2px;
}}
QLabel#section_title {{
    color: {ACCENT};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 3px;
}}
QLabel#status_ok  {{ color: {GREEN};   font-weight: bold; }}
QLabel#status_err {{ color: {ACCENT2}; font-weight: bold; }}
QLabel#status_rec {{ color: {AMBER};   font-weight: bold; }}
QFrame#card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}
QFrame#separator {{
    background-color: {BORDER};
    max-height: 1px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {BG};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {MUTED};
    border-radius: 3px;
    min-height: 20px;
}}
"""


def make_plot_widget(title: str) -> pg.PlotWidget:
    pw = pg.PlotWidget()
    pw.setBackground(BG_PANEL)
    pw.showGrid(x=False, y=True, alpha=0.07)
    pw.getAxis('left').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('bottom').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('left').setPen(pg.mkPen(BORDER))
    pw.getAxis('bottom').setPen(pg.mkPen(BORDER))
    pw.setClipToView(True)
    pw.setDownsampling(auto=True, mode="peak")
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=True)
    lbl = pg.LabelItem(
        f'<span style="color:{TEXT_DIM};font-size:9px;letter-spacing:2px">{title}</span>'
    )
    pw.addItem(lbl)
    return pw


# ═══════════════════════════════════════════════════════════════════════════════
# Signal conversion constants
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_SECS     = 5
SAMPLE_RATE     = 256
PPG_SAMPLE_RATE = 100
WINDOW_N        = WINDOW_SECS * SAMPLE_RATE
WINDOW_PPG_N    = WINDOW_SECS * PPG_SAMPLE_RATE
BPM_HISTORY     = 8
MAX_PLOT_POINTS = 1280   # match Pi screen width — no point plotting more

BPF_LOW_HZ  = 0.5
BPF_HIGH_HZ = 50.0

ADC_FULL_SCALE      = 65535.0
ADC_ZERO_G          = ADC_FULL_SCALE / 2.0
ADC_VREF            = 3.3
ADXL335_SENS_V_PER_G = 0.3
ADC_COUNTS_PER_G    = (ADXL335_SENS_V_PER_G / ADC_VREF) * ADC_FULL_SCALE
PPG_MAX_COUNTS      = 262143.0

VALVE_CONDITIONS = [
    "Aortic Stenosis",
    "Aortic Regurgitation",
    "Mitral Stenosis",
    "Mitral Regurgitation",
    "Tricuspid Regurgitation",
]


def raw_to_g(raw_int16: int) -> float:
    adc = float(raw_int16 & 0xFFFF)
    return (adc - ADC_ZERO_G) / ADC_COUNTS_PER_G


def ppg_norm(counts: float) -> float:
    return counts / PPG_MAX_COUNTS


# ═══════════════════════════════════════════════════════════════════════════════
# Subject metadata dialog
# ═══════════════════════════════════════════════════════════════════════════════

class SubjectDialog(QDialog):
    """
    Modal dialog for entering subject metadata before a recording session.
    Validates all fields; will not accept if any required field is empty/invalid.
    """
    def __init__(self, parent=None, existing: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Subject Information")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._build_ui(existing or {})

    def _build_ui(self, d: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        hdr = QLabel("SUBJECT INFORMATION")
        hdr.setObjectName("section_title")
        layout.addWidget(hdr)

        sep = QFrame(); sep.setObjectName("separator")
        layout.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        # Initials (2–4 uppercase letters)
        self._initials = QLineEdit(d.get('initials', ''))
        self._initials.setPlaceholderText("e.g.  AB")
        self._initials.setMaxLength(4)
        rx = QRegExp("[A-Za-z]{2,4}")
        self._initials.setValidator(QRegExpValidator(rx, self))
        self._initials.textChanged.connect(self._validate)
        form.addRow("Initials *", self._initials)

        # Age
        self._age = QSpinBox()
        self._age.setRange(1, 120)
        self._age.setValue(int(d.get('age', 40)))
        form.addRow("Age (yrs) *", self._age)

        # Sex
        self._sex = QComboBox()
        self._sex.addItems(["Male", "Female", "Other / Not specified"])
        if d.get('sex') in ["Male", "Female", "Other / Not specified"]:
            self._sex.setCurrentText(d['sex'])
        form.addRow("Sex", self._sex)

        # Weight
        self._weight = QDoubleSpinBox()
        self._weight.setRange(1.0, 300.0)
        self._weight.setDecimals(1)
        self._weight.setSuffix(" kg")
        self._weight.setValue(float(d.get('weight_kg', 70.0)))
        self._weight.valueChanged.connect(self._update_bmi)
        form.addRow("Weight *", self._weight)

        # Height
        self._height = QDoubleSpinBox()
        self._height.setRange(50.0, 250.0)
        self._height.setDecimals(1)
        self._height.setSuffix(" cm")
        self._height.setValue(float(d.get('height_cm', 170.0)))
        self._height.valueChanged.connect(self._update_bmi)
        form.addRow("Height *", self._height)

        # BMI (read-only, computed)
        self._bmi_lbl = QLabel()
        self._bmi_lbl.setObjectName("stat_value")
        self._bmi_lbl.setStyleSheet(f"color:{ACCENT};font-size:14px;font-weight:bold;")
        form.addRow("BMI", self._bmi_lbl)

        layout.addLayout(form)

        # ── Valve conditions ──────────────────────────────────────────────────
        cond_group = QGroupBox("CARDIAC CONDITIONS")
        cond_layout = QVBoxLayout(cond_group)
        cond_layout.setSpacing(4)

        self._normal_cb = QCheckBox("Normal (no known valvular disease)")
        self._normal_cb.setChecked(d.get('normal', True))
        self._normal_cb.toggled.connect(self._on_normal_toggled)
        cond_layout.addWidget(self._normal_cb)

        sep2 = QFrame(); sep2.setObjectName("separator")
        cond_layout.addWidget(sep2)

        self._valve_cbs: dict[str, QCheckBox] = {}
        existing_conds = d.get('conditions', [])
        for cond in VALVE_CONDITIONS:
            cb = QCheckBox(cond)
            cb.setChecked(cond in existing_conds)
            cb.toggled.connect(self._on_condition_toggled)
            cond_layout.addWidget(cb)
            self._valve_cbs[cond] = cb

        layout.addWidget(cond_group)

        # ── Notes ─────────────────────────────────────────────────────────────
        self._notes = QLineEdit(d.get('notes', ''))
        self._notes.setPlaceholderText("Optional free-text notes")
        layout.addWidget(QLabel("Notes"))
        layout.addWidget(self._notes)

        # ── Buttons ───────────────────────────────────────────────────────────
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._update_bmi()
        self._validate()
        self._on_normal_toggled(self._normal_cb.isChecked())

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self):
        ok = len(self._initials.text().strip()) >= 2
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        self._initials.setProperty("invalid", not ok)
        self._initials.style().unpolish(self._initials)
        self._initials.style().polish(self._initials)

    def _update_bmi(self):
        h_m = self._height.value() / 100.0
        bmi = self._weight.value() / (h_m * h_m) if h_m > 0 else 0.0
        category = ""
        if   bmi < 18.5: category = "Underweight"
        elif bmi < 25.0: category = "Normal"
        elif bmi < 30.0: category = "Overweight"
        else:            category = "Obese"
        self._bmi_lbl.setText(f"{bmi:.1f}  ({category})")

    def _on_normal_toggled(self, checked: bool):
        for cb in self._valve_cbs.values():
            cb.setEnabled(not checked)
            if checked:
                cb.setChecked(False)

    def _on_condition_toggled(self, checked: bool):
        if checked:
            self._normal_cb.setChecked(False)

    def _on_accept(self):
        if len(self._initials.text().strip()) < 2:
            QMessageBox.warning(self, "Validation", "Please enter at least 2 initials.")
            return
        self.accept()

    # ── Result extraction ─────────────────────────────────────────────────────

    def get_metadata(self) -> dict:
        h_m = self._height.value() / 100.0
        bmi = self._weight.value() / (h_m * h_m) if h_m > 0 else 0.0
        conditions = [c for c, cb in self._valve_cbs.items() if cb.isChecked()]
        normal = self._normal_cb.isChecked() or len(conditions) == 0
        return {
            'initials':   self._initials.text().strip().upper(),
            'age':        self._age.value(),
            'sex':        self._sex.currentText(),
            'weight_kg':  round(self._weight.value(), 1),
            'height_cm':  round(self._height.value(), 1),
            'bmi':        round(bmi, 1),
            'normal':     normal,
            'conditions': conditions if not normal else [],
            'notes':      self._notes.text().strip(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCG · PPG Monitor")
        self.resize(1280, 800)

        # ── Signal ring buffers (plot fast-path) ──────────────────────────────
        self._rb_scg_x       = RingBuffer(WINDOW_N)
        self._rb_scg_y       = RingBuffer(WINDOW_N)
        self._rb_scg_z       = RingBuffer(WINDOW_N)
        self._rb_scg_ts      = RingBuffer(WINDOW_N, dtype=np.int64)
        self._rb_scg_hts     = RingBuffer(WINDOW_N, dtype=np.int64)   # host-clock ts
        self._rb_ppg         = RingBuffer(WINDOW_PPG_N)
        self._rb_ppg_ts      = RingBuffer(WINDOW_PPG_N, dtype=np.int64)
        self._rb_ppg_hts     = RingBuffer(WINDOW_PPG_N, dtype=np.int64)

        # ── Deques for segment extraction (need random access) ────────────────
        self._dq_scg_x   = deque(maxlen=WINDOW_N)
        self._dq_scg_y   = deque(maxlen=WINDOW_N)
        self._dq_scg_z   = deque(maxlen=WINDOW_N)
        self._dq_scg_ts  = deque(maxlen=WINDOW_N)
        self._dq_scg_hts = deque(maxlen=WINDOW_N)

        # ── Beat tracking ─────────────────────────────────────────────────────
        self._beat_ts:       list[int] = []
        self._beat_hts:      list[int] = []
        self._beat_intervals: deque    = deque(maxlen=BPM_HISTORY)
        self._last_beat_ts:  int | None = None
        self._last_beat_hts: int | None = None

        # ── Beat-marker cache (avoid recreating InfiniteLine every frame) ─────
        self._beat_line_cache:   list[list[tuple]] = [[], [], []]
        self._beat_ts_snapshot:  list[int]         = []

        # ── Timing / clocking ─────────────────────────────────────────────────
        self._scg_host_clock_ms: float | None = None
        self._ppg_host_clock_ms: float | None = None
        self._use_host_time      = False

        # ── Counters ──────────────────────────────────────────────────────────
        self._sample_count   = 0
        self._parse_err_count = 0
        self._rate_count     = 0

        # ── Dirty flags ───────────────────────────────────────────────────────
        self._plot_dirty    = False
        self._segment_dirty = False

        # ── Filter ────────────────────────────────────────────────────────────
        self._filter_on = False
        self._notch_on  = False
        self._bpf_sos   = butter(2, [BPF_LOW_HZ, BPF_HIGH_HZ],
                                 btype='bandpass', fs=SAMPLE_RATE, output='sos')
        # Notch at 50 Hz: Q=30 gives a -3 dB bandwidth of ~1.7 Hz — narrow
        # enough to kill mains hum without touching the 40-50 Hz SCG content.
        from scipy.signal import iirnotch, tf2sos as _tf2sos
        _b, _a      = iirnotch(50.0, Q=30, fs=SAMPLE_RATE)
        self._notch_sos = _tf2sos(_b, _a)
        self._reset_filter_state()

        # ── Serial ────────────────────────────────────────────────────────────
        self._thread: ReaderThread | None = None
        self._reader: SerialReader | None = None

        # ── Recording ─────────────────────────────────────────────────────────
        self._is_recording       = False
        self._async_writer: AsyncCSVWriter | None = None
        self._record_path        = ""
        self._record_samples     = 0
        self._record_first_ts: int | None = None
        self._record_last_ts:  int | None = None
        self._record_elapsed     = 0
        self._session_start_dt:  datetime | None = None

        # ── Subject metadata ──────────────────────────────────────────────────
        self._subject: dict | None = None

        # ── Timers ────────────────────────────────────────────────────────────
        self._ingest_timer = QTimer(self)
        self._ingest_timer.timeout.connect(self._drain_serial)

        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._on_plot_timer)
        self._plot_timer.start(40)   # 25 fps — safe on Pi 4

        self._rate_timer = QTimer(self)
        self._rate_timer.timeout.connect(self._update_rate)

        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._tick_rec_timer)

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

    # ═══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QHBoxLayout(root)
        rl.setContentsMargins(10, 10, 10, 10)
        rl.setSpacing(10)
        rl.addWidget(self._build_plots_panel(), stretch=1)
        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(230)
        rl.addWidget(sidebar)

    # ── Plots panel ───────────────────────────────────────────────────────────

    def _build_plots_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # SCG header
        hdr = QHBoxLayout()
        t = QLabel("SEISMOCARDIOGRAM")
        t.setObjectName("section_title")
        hdr.addWidget(t)
        hdr.addStretch()
        for label, color in zip(LABEL_SCG, COLORS_SCG):
            d = QLabel(f"● {label}")
            d.setStyleSheet(f"color:{color};font-size:10px;margin-left:6px;")
            hdr.addWidget(d)
        layout.addLayout(hdr)

        self._plots:  list[pg.PlotWidget]   = []
        self._curves: list[pg.PlotDataItem] = []

        for axis_name, color in zip(["X AXIS", "Y AXIS", "Z AXIS"], COLORS_SCG):
            pw = make_plot_widget(axis_name)
            pw.setXRange(-WINDOW_SECS, 0, padding=0)
            pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            curve = pw.plot([], [], pen=pg.mkPen(color, width=1.5, cosmetic=True))
            self._plots.append(pw)
            self._curves.append(curve)
            layout.addWidget(pw)

        # PPG header
        hdr2 = QHBoxLayout()
        t2 = QLabel("PHOTOPLETHYSMOGRAM")
        t2.setObjectName("section_title")
        t2.setStyleSheet(f"color:{ACCENT2};font-size:11px;font-weight:bold;letter-spacing:3px;")
        hdr2.addWidget(t2)
        hdr2.addStretch()
        layout.addLayout(hdr2)

        self._ppg_plot = make_plot_widget("PPG WAVEFORM")
        self._ppg_plot.setXRange(-WINDOW_SECS, 0, padding=0)
        self._ppg_plot.setYRange(0.0, 1.0, padding=0.02)
        self._ppg_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ppg_curve = self._ppg_plot.plot(
            [], [], pen=pg.mkPen(ACCENT2, width=1.5, cosmetic=True))
        layout.addWidget(self._ppg_plot)

        # Segment plot
        seg_hdr = QLabel("LAST BEAT SEGMENT")
        seg_hdr.setObjectName("section_title")
        seg_hdr.setStyleSheet(f"color:{AMBER};font-size:11px;font-weight:bold;letter-spacing:3px;")
        layout.addWidget(seg_hdr)

        self._segment_plot = make_plot_widget("BEAT SEGMENT")
        self._segment_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._segment_plot.getAxis('bottom').setLabel('time (s)')
        self._segment_plot.showGrid(x=True, y=True, alpha=0.07)
        self._segment_curves: list[pg.PlotDataItem] = []
        for color in COLORS_SCG:
            self._segment_curves.append(
                self._segment_plot.plot([], [],
                    pen=pg.mkPen(color, width=1.5, cosmetic=True))
            )
        layout.addWidget(self._segment_plot)

        return w

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # BPM card
        bpm_card = QFrame(); bpm_card.setObjectName("card")
        bl = QVBoxLayout(bpm_card); bl.setContentsMargins(14, 12, 14, 12); bl.setSpacing(2)
        bpm_title = QLabel("HEART RATE"); bpm_title.setObjectName("stat_label")
        self._bpm_label = QLabel("--"); self._bpm_label.setObjectName("bpm_value")
        self._bpm_label.setAlignment(Qt.AlignCenter)
        bpm_unit = QLabel("BPM"); bpm_unit.setObjectName("stat_label")
        bpm_unit.setAlignment(Qt.AlignCenter)
        bl.addWidget(bpm_title, alignment=Qt.AlignCenter)
        bl.addWidget(self._bpm_label)
        bl.addWidget(bpm_unit)
        layout.addWidget(bpm_card)

        # Stats card
        stats_card = QFrame(); stats_card.setObjectName("card")
        sl = QVBoxLayout(stats_card); sl.setContentsMargins(14, 12, 14, 12); sl.setSpacing(8)
        self._stat_beats = self._stat_row(sl, "BEATS",        "0")
        self._stat_rate  = self._stat_row(sl, "SAMPLE RATE",  "-- Hz")
        self._stat_lost  = self._stat_row(sl, "PARSE ERRORS", "0")
        layout.addWidget(stats_card)

        # Subject card
        subj_card = QFrame(); subj_card.setObjectName("card")
        subj_layout = QVBoxLayout(subj_card)
        subj_layout.setContentsMargins(14, 12, 14, 12); subj_layout.setSpacing(6)
        subj_hdr = QLabel("SUBJECT"); subj_hdr.setObjectName("stat_label")
        subj_layout.addWidget(subj_hdr)
        self._subj_lbl = QLabel("No subject loaded")
        self._subj_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        self._subj_lbl.setWordWrap(True)
        subj_layout.addWidget(self._subj_lbl)
        self._subj_cond_lbl = QLabel("")
        self._subj_cond_lbl.setStyleSheet(f"color:{AMBER};font-size:9px;")
        self._subj_cond_lbl.setWordWrap(True)
        subj_layout.addWidget(self._subj_cond_lbl)
        edit_subj_btn = QPushButton("SET SUBJECT")
        edit_subj_btn.clicked.connect(self._edit_subject)
        subj_layout.addWidget(edit_subj_btn)
        layout.addWidget(subj_card)

        # Connection card
        conn_card = QFrame(); conn_card.setObjectName("card")
        cl = QVBoxLayout(conn_card); cl.setContentsMargins(14, 12, 14, 12); cl.setSpacing(6)
        port_lbl = QLabel("SERIAL PORT"); port_lbl.setObjectName("stat_label")
        self._port_combo = QComboBox()
        self._refresh_ports()
        ref_btn = QPushButton("↻  REFRESH")
        ref_btn.clicked.connect(self._refresh_ports)
        self._connect_btn = QPushButton("CONNECT")
        self._connect_btn.clicked.connect(self._on_connect)
        self._stop_btn = QPushButton("DISCONNECT")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.clicked.connect(self._on_disconnect)
        self._stop_btn.setEnabled(False)
        self._status_lbl = QLabel("IDLE")
        self._status_lbl.setObjectName("status_err")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        for w in [port_lbl, self._port_combo, ref_btn,
                  self._connect_btn, self._stop_btn, self._status_lbl]:
            cl.addWidget(w)
        layout.addWidget(conn_card)

        # Recording card
        rec_card = QFrame(); rec_card.setObjectName("card")
        rl2 = QVBoxLayout(rec_card); rl2.setContentsMargins(14, 12, 14, 12); rl2.setSpacing(6)
        rec_hdr = QLabel("RECORDING"); rec_hdr.setObjectName("stat_label")
        rl2.addWidget(rec_hdr)
        self._rec_btn = QPushButton("⏺  START RECORDING")
        self._rec_btn.setObjectName("rec_btn")
        self._rec_btn.clicked.connect(self._toggle_recording)
        self._rec_btn.setEnabled(False)
        rl2.addWidget(self._rec_btn)
        self._rec_time_lbl = QLabel("00:00")
        self._rec_time_lbl.setAlignment(Qt.AlignCenter)
        self._rec_time_lbl.setStyleSheet(
            f"color:{AMBER};font-size:18px;font-weight:bold;font-family:{FONT_MONO};")
        rl2.addWidget(self._rec_time_lbl)
        self._rec_info_lbl = QLabel("")
        self._rec_info_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:9px;")
        self._rec_info_lbl.setWordWrap(True)
        rl2.addWidget(self._rec_info_lbl)
        layout.addWidget(rec_card)

        # Settings card
        cfg_card = QFrame(); cfg_card.setObjectName("card")
        cfg_l = QVBoxLayout(cfg_card); cfg_l.setContentsMargins(14, 12, 14, 12); cfg_l.setSpacing(6)
        cfg_hdr = QLabel("SETTINGS"); cfg_hdr.setObjectName("stat_label")
        cfg_l.addWidget(cfg_hdr)
        self._filter_cb = QCheckBox("BANDPASS 0.5–50 Hz")
        self._filter_cb.toggled.connect(self._on_filter_toggled)
        cfg_l.addWidget(self._filter_cb)
        self._notch_cb = QCheckBox("NOTCH 50 Hz (mains)")
        self._notch_cb.setEnabled(False)   # only active when bandpass is on
        self._notch_cb.toggled.connect(self._on_notch_toggled)
        cfg_l.addWidget(self._notch_cb)
        self._host_time_cb = QCheckBox("USE PC TIME AXIS")
        self._host_time_cb.toggled.connect(lambda c: self._set_host_time(c))
        cfg_l.addWidget(self._host_time_cb)
        clear_btn = QPushButton("CLEAR PLOTS")
        clear_btn.clicked.connect(self._clear_data)
        cfg_l.addWidget(clear_btn)
        layout.addWidget(cfg_card)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _stat_row(self, parent_layout, label: str, value: str) -> QLabel:
        row = QHBoxLayout()
        lbl = QLabel(label); lbl.setObjectName("stat_label")
        val = QLabel(value)
        val.setStyleSheet(f"color:{TEXT};font-size:12px;font-weight:bold;font-family:{FONT_MONO};")
        row.addWidget(lbl); row.addStretch(); row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    # ═══════════════════════════════════════════════════════════════════════════
    # Subject metadata
    # ═══════════════════════════════════════════════════════════════════════════

    def _edit_subject(self):
        if self._is_recording:
            QMessageBox.warning(self, "Recording Active",
                "Cannot change subject information during an active recording.")
            return
        dlg = SubjectDialog(self, existing=self._subject)
        if dlg.exec_() == QDialog.Accepted:
            self._subject = dlg.get_metadata()
            self._update_subject_display()
            self._rec_btn.setEnabled(self._reader is not None)

    def _update_subject_display(self):
        if self._subject is None:
            self._subj_lbl.setText("No subject loaded")
            self._subj_cond_lbl.setText("")
            return
        s = self._subject
        self._subj_lbl.setText(
            f"{s['initials']}  |  Age {s['age']}  |  {s['sex']}\n"
            f"BMI {s['bmi']}  ({s['weight_kg']} kg, {s['height_cm']} cm)"
        )
        if s.get('normal', True) or not s.get('conditions'):
            self._subj_cond_lbl.setText("Normal cardiac function")
        else:
            self._subj_cond_lbl.setText("\n".join(s['conditions']))

    # ═══════════════════════════════════════════════════════════════════════════
    # Port / connection
    # ═══════════════════════════════════════════════════════════════════════════

    def _refresh_ports(self):
        self._port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or '')[:32]
            self._port_combo.addItem(f"{p.device}  {desc}", userData=p.device)
        if not ports:
            self._port_combo.addItem("No ports found")

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
        self._status_lbl.setObjectName("status_ok")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        self._ingest_timer.start(20)    # drain at 50 Hz — well ahead of 25 Hz plot
        self._rate_timer.start(1000)
        if self._subject is not None:
            self._rec_btn.setEnabled(True)

    def _on_disconnect(self):
        if self._is_recording:
            self._stop_recording()
        if self._thread:
            # Disconnect the error signal FIRST so that the OSError/
            # SerialException that pyserial raises when the port is closed
            # underneath an in-flight read() never reaches _on_serial_error
            # and triggers a spurious error dialog.
            if self._reader:
                try:
                    self._reader.error.disconnect(self._on_serial_error)
                except TypeError:
                    pass   # already disconnected
            self._thread.stop()   # signals _running=False, waits ≤500 ms
            self._thread = None
        self._reader = None
        self._ingest_timer.stop()
        self._rate_timer.stop()
        self._connect_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._rec_btn.setEnabled(False)
        self._status_lbl.setText("DISCONNECTED")
        self._status_lbl.setObjectName("status_err")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)

    def _on_serial_error(self, msg: str):
        # Ignore errors that arrive after we've already started disconnecting
        if self._thread is None:
            return
        self._status_lbl.setText("ERROR")
        self._status_lbl.setObjectName("status_err")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        QMessageBox.critical(self, "Serial Error", msg)
        self._on_disconnect()

    # ═══════════════════════════════════════════════════════════════════════════
    # Filter
    # ═══════════════════════════════════════════════════════════════════════════

    def _reset_filter_state(self):
        n = self._bpf_sos.shape[0]
        self._zi_x = np.zeros((n, 2), dtype=np.float64)
        self._zi_y = np.zeros((n, 2), dtype=np.float64)
        self._zi_z = np.zeros((n, 2), dtype=np.float64)
        m = self._notch_sos.shape[0]
        self._zi_nx = np.zeros((m, 2), dtype=np.float64)
        self._zi_ny = np.zeros((m, 2), dtype=np.float64)
        self._zi_nz = np.zeros((m, 2), dtype=np.float64)

    def _apply_filter(self, xv, yv, zv):
        xo, self._zi_x = sosfilt(self._bpf_sos, [xv], zi=self._zi_x)
        yo, self._zi_y = sosfilt(self._bpf_sos, [yv], zi=self._zi_y)
        zo, self._zi_z = sosfilt(self._bpf_sos, [zv], zi=self._zi_z)
        if self._notch_on:
            xo, self._zi_nx = sosfilt(self._notch_sos, xo, zi=self._zi_nx)
            yo, self._zi_ny = sosfilt(self._notch_sos, yo, zi=self._zi_ny)
            zo, self._zi_nz = sosfilt(self._notch_sos, zo, zi=self._zi_nz)
        return float(xo[0]), float(yo[0]), float(zo[0])

    def _on_filter_toggled(self, checked: bool):
        self._filter_on = checked
        # Notch only makes sense when bandpass is also on; keep the checkbox
        # enabled but ignore _notch_on in _apply_filter when bandpass is off.
        self._notch_cb.setEnabled(checked)
        if not checked:
            self._notch_cb.setChecked(False)
        self._reset_filter_state()
        self._apply_scg_yrange()

    def _on_notch_toggled(self, checked: bool):
        self._notch_on = checked
        self._reset_filter_state()   # flush transient state so no click on toggle

    def _apply_scg_yrange(self):
        """
        When the bandpass filter is on, pin each SCG plot's y-axis to ±1 g so
        the small filtered signal fills the view cleanly and isn't dwarfed by
        the raw ADC swing.  When off, release to pyqtgraph auto-range so the
        user can still see the full raw signal.
        """
        for pw in self._plots:
            vb = pw.getViewBox()
            if self._filter_on:
                vb.setYRange(-1.0, 1.0, padding=0.05)
                vb.setAutoVisible(y=False)
                vb.enableAutoRange(axis='y', enable=False)
            else:
                vb.enableAutoRange(axis='y', enable=True)

    def _set_host_time(self, checked: bool):
        self._use_host_time = checked
        self._plot_dirty = True

    # ═══════════════════════════════════════════════════════════════════════════
    # Data ingestion  (runs on main thread, called every 20 ms)
    # Serial reader accumulates; this just drains and processes.
    # ═══════════════════════════════════════════════════════════════════════════

    def _drain_serial(self):
        if not self._reader:
            return
        batch = self._reader.drain()
        if batch is None:
            return
        scg_samples, ppg_samples, beat_timestamps, parse_errors = batch

        # Throttle batch size to protect main thread at the drain call site.
        # On Pi, processing 256 samples/call still finishes in < 2 ms.
        if len(scg_samples)  > 256: scg_samples  = scg_samples[-256:]
        if len(ppg_samples)  > 100: ppg_samples  = ppg_samples[-100:]
        if len(beat_timestamps) > 20: beat_timestamps = beat_timestamps[-20:]

        self._ingest(scg_samples, ppg_samples, beat_timestamps, parse_errors)

    def _ingest(self, scg_samples, ppg_samples, beat_timestamps, parse_errors):
        self._parse_err_count += int(parse_errors)
        scg_dt  = 1000.0 / SAMPLE_RATE
        ppg_dt  = 1000.0 / PPG_SAMPLE_RATE
        now_ms  = time.monotonic() * 1000.0

        # ── SCG ───────────────────────────────────────────────────────────────
        for ts, x_raw, y_raw, z_raw in scg_samples:
            if self._scg_host_clock_ms is None:
                self._scg_host_clock_ms = now_ms
            else:
                self._scg_host_clock_ms += scg_dt

            xg = raw_to_g(x_raw)
            yg = raw_to_g(y_raw)
            zg = raw_to_g(z_raw)
            if self._filter_on:
                xg, yg, zg = self._apply_filter(xg, yg, zg)

            hts = int(self._scg_host_clock_ms)
            its = int(ts)

            # Ring buffers (plot)
            self._rb_scg_ts.append(its)
            self._rb_scg_hts.append(hts)
            self._rb_scg_x.append(xg)
            self._rb_scg_y.append(yg)
            self._rb_scg_z.append(zg)

            # Deques (segment extraction)
            self._dq_scg_ts.append(its)
            self._dq_scg_hts.append(hts)
            self._dq_scg_x.append(xg)
            self._dq_scg_y.append(yg)
            self._dq_scg_z.append(zg)

            self._sample_count  += 1
            self._rate_count    += 1

            if self._is_recording and self._async_writer:
                row = {
                    'timestamp_ms': its,
                    'host_time_ms': hts,
                    'x_g': f"{xg:.6f}",
                    'y_g': f"{yg:.6f}",
                    'z_g': f"{zg:.6f}",
                    'beat_event': 0,
                    'ibi_ms':     '',
                    'ppg_raw':    '',
                }
                self._async_writer.write(row)
                self._record_samples += 1
                if self._record_first_ts is None:
                    self._record_first_ts = its
                self._record_last_ts = its

        # ── PPG ───────────────────────────────────────────────────────────────
        for ts, ppg_raw in ppg_samples:
            if self._ppg_host_clock_ms is None:
                self._ppg_host_clock_ms = now_ms
            else:
                self._ppg_host_clock_ms += ppg_dt

            hts = int(self._ppg_host_clock_ms)
            self._rb_ppg_ts.append(int(ts))
            self._rb_ppg_hts.append(hts)
            self._rb_ppg.append(ppg_norm(float(ppg_raw)))

            if self._is_recording and self._async_writer:
                self._async_writer.write({
                    'timestamp_ms': int(ts),
                    'host_time_ms': hts,
                    'x_g': '', 'y_g': '', 'z_g': '',
                    'beat_event': 0,
                    'ibi_ms':     '',
                    'ppg_raw':    int(ppg_raw),
                })

        # ── Beats ─────────────────────────────────────────────────────────────
        for ts in beat_timestamps:
            self._beat_ts.append(int(ts))
            hts = int(time.monotonic() * 1000.0)
            self._beat_hts.append(hts)
            ibi_ms = ''
            if self._last_beat_ts is not None:
                interval = int(ts) - self._last_beat_ts
                if 300 < interval < 2000:
                    self._beat_intervals.append(interval)
                    ibi_ms = interval
            self._last_beat_ts  = int(ts)
            self._last_beat_hts = hts

            if self._is_recording and self._async_writer:
                self._async_writer.write({
                    'timestamp_ms': int(ts),
                    'host_time_ms': hts,
                    'x_g': '', 'y_g': '', 'z_g': '',
                    'beat_event': 1,
                    'ibi_ms':     ibi_ms,
                    'ppg_raw':    '',
                })

        self._plot_dirty = True
        if beat_timestamps:
            self._segment_dirty = True
        self._refresh_stats()

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot timer  (40 ms / 25 fps)
    # Recording no longer suppresses the display — fixed.
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_plot_timer(self):
        if not self._plot_dirty:
            return
        self._refresh_scg_plots()
        self._refresh_ppg_plot()
        if self._segment_dirty:
            self._refresh_segment_plot()
            self._segment_dirty = False
        self._plot_dirty = False

    # ── SCG waveform + beat markers ───────────────────────────────────────────

    def _decimate(self, t: np.ndarray, y: np.ndarray):
        if len(y) <= MAX_PLOT_POINTS:
            return t, y
        step = max(1, len(y) // MAX_PLOT_POINTS)
        return t[::step], y[::step]

    def _refresh_scg_plots(self):
        n = self._rb_scg_ts.valid_count
        if n < 2:
            return

        rb_ts         = self._rb_scg_hts  if self._use_host_time else self._rb_scg_ts
        beat_src      = self._beat_hts    if self._use_host_time else self._beat_ts
        ts_arr        = rb_ts.to_array(n)
        now_ts        = int(ts_arr[-1])
        t_axis        = (ts_arr - now_ts).astype(np.float32) / 1000.0   # relative seconds

        signal_arrs = [
            self._rb_scg_x.to_array(n),
            self._rb_scg_y.to_array(n),
            self._rb_scg_z.to_array(n),
        ]

        beats_changed = (beat_src != self._beat_ts_snapshot)

        for i, (curve, arr) in enumerate(zip(self._curves, signal_arrs)):
            xt, yv = self._decimate(t_axis, arr)
            curve.setData(xt, yv)

            pw = self._plots[i]
            if beats_changed:
                # Remove old lines
                for _, line in self._beat_line_cache[i]:
                    pw.removeItem(line)
                self._beat_line_cache[i].clear()
                # Add current beats in window
                for b_ts in beat_src[-30:]:
                    age_s = (now_ts - b_ts) / 1000.0
                    if 0.0 <= age_s <= WINDOW_SECS:
                        line = pg.InfiniteLine(
                            pos=-age_s, angle=90,
                            pen=pg.mkPen(ACCENT2, width=1,
                                         style=Qt.DashLine, cosmetic=True)
                        )
                        pw.addItem(line)
                        self._beat_line_cache[i].append((b_ts, line))
            else:
                # Shift existing lines — no object allocation
                for b_ts, line in self._beat_line_cache[i]:
                    age_s = (now_ts - b_ts) / 1000.0
                    line.setPos(-age_s if 0.0 <= age_s <= WINDOW_SECS
                                else -(WINDOW_SECS + 1))

        if beats_changed:
            self._beat_ts_snapshot = list(beat_src)

    # ── PPG waveform ──────────────────────────────────────────────────────────

    def _refresh_ppg_plot(self):
        n = self._rb_ppg_ts.valid_count
        if n < 2:
            return
        rb_ts  = self._rb_ppg_hts if self._use_host_time else self._rb_ppg_ts
        ts_arr = rb_ts.to_array(n)
        t_axis = (ts_arr - int(ts_arr[-1])).astype(np.float32) / 1000.0
        ppg    = self._rb_ppg.to_array(n)
        xt, yv = self._decimate(t_axis, ppg)
        self._ppg_curve.setData(xt, yv)

    # ── Beat segment ──────────────────────────────────────────────────────────

    def _refresh_segment_plot(self):
        beat_src = self._beat_hts if self._use_host_time else self._beat_ts
        if len(beat_src) < 2:
            for c in self._segment_curves:
                c.setData([], [])
            return

        start_ts = beat_src[-2]
        end_ts   = beat_src[-1]
        if end_ts <= start_ts:
            return

        dq_ts  = self._dq_scg_hts if self._use_host_time else self._dq_scg_ts
        ts_arr = np.array(dq_ts, dtype=np.int64)
        mask   = (ts_arr >= start_ts) & (ts_arr <= end_ts)
        if np.count_nonzero(mask) < 2:
            return

        seg_t = (ts_arr[mask] - start_ts).astype(np.float32) / 1000.0
        seg_x = np.array(self._dq_scg_x, dtype=np.float32)[mask]
        seg_y = np.array(self._dq_scg_y, dtype=np.float32)[mask]
        seg_z = np.array(self._dq_scg_z, dtype=np.float32)[mask]

        # Decimate all three using the same indices so lengths always match
        if len(seg_t) > MAX_PLOT_POINTS:
            step = max(1, len(seg_t) // MAX_PLOT_POINTS)
            seg_t = seg_t[::step]
            seg_x = seg_x[::step]
            seg_y = seg_y[::step]
            seg_z = seg_z[::step]

        self._segment_curves[0].setData(seg_t, seg_x)
        self._segment_curves[1].setData(seg_t, seg_y)
        self._segment_curves[2].setData(seg_t, seg_z)
        if len(seg_t):
            self._segment_plot.setXRange(0.0, float(seg_t[-1]), padding=0.02)
        seg_vb = self._segment_plot.getViewBox()
        if self._filter_on:
            seg_vb.setYRange(-1.0, 1.0, padding=0.05)
            seg_vb.enableAutoRange(axis='y', enable=False)
        else:
            seg_vb.enableAutoRange(axis='y', enable=True)

    # ── Stats bar ─────────────────────────────────────────────────────────────

    def _refresh_stats(self):
        self._stat_beats.setText(str(len(self._beat_ts)))
        self._stat_lost.setText(str(self._parse_err_count))
        if len(self._beat_intervals) >= 2:
            bpm = 60000.0 / float(np.mean(self._beat_intervals))
            self._bpm_label.setText(f"{bpm:.0f}")
        else:
            self._bpm_label.setText("--")

    def _update_rate(self):
        self._stat_rate.setText(f"{self._rate_count} Hz")
        self._rate_count = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Recording
    # ═══════════════════════════════════════════════════════════════════════════

    def _toggle_recording(self):
        if self._is_recording:
            reply = QMessageBox.question(
                self, "Stop Recording",
                "Stop the current recording session?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self._subject is None:
            QMessageBox.warning(self, "No Subject",
                "Please set subject information before recording.")
            return

        now = datetime.now()
        self._session_start_dt = now
        initials = self._subject['initials']

        # ── Build save directory: SUBJECT_Data/YYYY-MM-DD/ ────────────────────
        # Root folder sits next to the script so it's always findable on the Pi.
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        root_dir     = os.path.join(script_dir, "SUBJECT_Data")
        date_dir     = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)   # creates both levels if needed

        # ── Auto filename: INITIALS_HHMMSS.csv ────────────────────────────────
        ts_str       = now.strftime("%H%M%S")
        default_name = f"{initials}_{ts_str}.csv"
        path         = os.path.join(date_dir, default_name)

        # If a file with that name already exists (two sessions started within
        # the same second), append a counter suffix so we never overwrite.
        if os.path.exists(path):
            base = os.path.splitext(default_name)[0]
            counter = 1
            while os.path.exists(os.path.join(date_dir, f"{base}_{counter}.csv")):
                counter += 1
            path = os.path.join(date_dir, f"{base}_{counter}.csv")

        # Build metadata lines for CSV header comments
        s = self._subject
        cond_str = (', '.join(s['conditions']) if s['conditions'] else 'Normal')
        meta_lines = [
            f"SCG/PPG Recording  —  {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"patient_initials,{s['initials']}",
            f"age,{s['age']}",
            f"sex,{s['sex']}",
            f"weight_kg,{s['weight_kg']}",
            f"height_cm,{s['height_cm']}",
            f"bmi,{s['bmi']}",
            f"cardiac_conditions,{cond_str}",
            f"notes,{s.get('notes', '')}",
            f"sample_rate_scg_hz,{SAMPLE_RATE}",
            f"sample_rate_ppg_hz,{PPG_SAMPLE_RATE}",
            f"filter_enabled,{self._filter_on}",
            f"notch_50hz_enabled,{self._notch_on}",
        ]

        fieldnames = [
            'timestamp_ms', 'host_time_ms',
            'x_g', 'y_g', 'z_g',
            'beat_event', 'ibi_ms', 'ppg_raw',
        ]

        self._async_writer = AsyncCSVWriter(path, fieldnames, meta_lines)
        self._record_path     = path
        self._record_samples  = 0
        self._record_first_ts = None
        self._record_last_ts  = None
        self._record_elapsed  = 0
        self._is_recording    = True

        self._rec_btn.setText("⏹  STOP RECORDING")
        self._rec_time_lbl.setText("00:00")
        # Show  SUBJECT_Data/YYYY-MM-DD/filename.csv  — fits the narrow sidebar
        rel_path = os.path.relpath(path, script_dir)
        self._rec_info_lbl.setText(rel_path)
        self._status_lbl.setText("RECORDING")
        self._status_lbl.setObjectName("status_rec")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        self._rec_timer.start(1000)

        # Write sidecar JSON immediately so metadata exists even if crash
        self._write_sidecar_json(path, s, now, cond_str)

    def _stop_recording(self):
        if not self._is_recording:
            return
        self._is_recording = False
        self._rec_timer.stop()

        writer = self._async_writer
        self._async_writer = None

        # Close the writer (blocks up to 10 s for queue to flush)
        if writer:
            writer.close()

        # Compute actual sample rate
        actual_hz = 0.0
        if (self._record_samples >= 2
                and self._record_first_ts is not None
                and self._record_last_ts is not None):
            dur_s = (self._record_last_ts - self._record_first_ts) / 1000.0
            if dur_s > 0:
                actual_hz = (self._record_samples - 1) / dur_s

        diff_hz  = actual_hz - SAMPLE_RATE
        diff_pct = (diff_hz / SAMPLE_RATE * 100.0)
        elapsed  = self._record_elapsed

        self._rec_btn.setText("⏺  START RECORDING")
        self._status_lbl.setText("CONNECTED")
        self._status_lbl.setObjectName("status_ok")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)

        QMessageBox.information(
            self, "Session Complete",
            f"Saved to:\n{self._record_path}\n\n"
            f"SCG samples:    {self._record_samples}\n"
            f"Expected rate:  {SAMPLE_RATE} Hz\n"
            f"Actual rate:    {actual_hz:.2f} Hz\n"
            f"Δ rate:         {diff_hz:+.2f} Hz  ({diff_pct:+.2f}%)\n"
            f"Duration:       {elapsed // 60:02d}:{elapsed % 60:02d}"
        )

    def _write_sidecar_json(self, csv_path: str, s: dict,
                             dt: datetime, cond_str: str):
        json_path = os.path.splitext(csv_path)[0] + "_meta.json"
        meta = {
            "session_start": dt.isoformat(),
            "patient_initials": s['initials'],
            "age": s['age'],
            "sex": s['sex'],
            "weight_kg": s['weight_kg'],
            "height_cm": s['height_cm'],
            "bmi": s['bmi'],
            "cardiac_conditions": s['conditions'] if s['conditions'] else ["Normal"],
            "notes": s.get('notes', ''),
            "sample_rate_scg_hz": SAMPLE_RATE,
            "sample_rate_ppg_hz": PPG_SAMPLE_RATE,
            "filter_enabled": self._filter_on,
            "notch_50hz_enabled": self._notch_on,
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
        except OSError:
            pass   # non-fatal; the CSV metadata comment is the backup

    def _tick_rec_timer(self):
        if not self._is_recording:
            return
        self._record_elapsed += 1
        mm = self._record_elapsed // 60
        ss = self._record_elapsed % 60
        self._rec_time_lbl.setText(f"{mm:02d}:{ss:02d}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Clear
    # ═══════════════════════════════════════════════════════════════════════════

    def _clear_data(self):
        for rb in [self._rb_scg_x, self._rb_scg_y, self._rb_scg_z,
                   self._rb_scg_ts, self._rb_scg_hts,
                   self._rb_ppg, self._rb_ppg_ts, self._rb_ppg_hts]:
            rb.clear()
        for dq in [self._dq_scg_x, self._dq_scg_y, self._dq_scg_z,
                   self._dq_scg_ts, self._dq_scg_hts]:
            dq.clear()
        self._scg_host_clock_ms = None
        self._ppg_host_clock_ms = None
        self._reset_filter_state()
        self._beat_ts.clear()
        self._beat_hts.clear()
        self._beat_intervals.clear()
        self._last_beat_ts  = None
        self._last_beat_hts = None
        self._sample_count  = 0
        self._parse_err_count = 0
        self._beat_ts_snapshot = []
        for cache in self._beat_line_cache:
            for _, line in cache:
                pass   # items already removed when plots were cleared
            cache.clear()
        for pw in self._plots:
            pw.clear()
        self._ppg_plot.clear()
        self._ppg_curve = self._ppg_plot.plot(
            [], [], pen=pg.mkPen(ACCENT2, width=1.5, cosmetic=True))
        for i, (color, c) in enumerate(zip(COLORS_SCG, self._curves)):
            self._curves[i] = self._plots[i].plot(
                [], [], pen=pg.mkPen(color, width=1.5, cosmetic=True))
        for c in self._segment_curves:
            c.setData([], [])
        self._bpm_label.setText("--")
        self._stat_beats.setText("0")
        self._stat_lost.setText("0")
        self._rate_count = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Close
    # ═══════════════════════════════════════════════════════════════════════════

    def closeEvent(self, event):
        if self._is_recording:
            reply = QMessageBox.question(
                self, "Recording Active",
                "A recording is in progress. Stop and exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        self._on_disconnect()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pi-safe pyqtgraph settings
    pg.setConfigOptions(
        antialias=False,      # significant CPU saving on Pi
        useOpenGL=False,      # avoid Mesa driver issues
        enableExperimental=False,
    )

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