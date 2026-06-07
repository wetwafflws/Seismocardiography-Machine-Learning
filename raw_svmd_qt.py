from __future__ import annotations

import csv
import json
import os
import queue
import serial
import serial.tools.list_ports
import struct
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QRegExp
from PyQt5.QtGui import QColor, QPalette, QRegExpValidator, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QFrame,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import signal
from scipy.signal import hilbert, savgol_filter, butter, sosfilt
from scipy.stats import kurtosis


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
    scg_samples     = []
    ppg_samples     = []
    beat_timestamps = []
    parse_errors    = 0

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
# Serial Reader
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

        self._lock         = threading.Lock()
        self._scg_accum:   list = []
        self._ppg_accum:   list = []
        self._beat_accum:  list = []
        self._parse_errs:  int  = 0

    def start(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        except serial.SerialException as e:
            self.error.emit(str(e))
            return
        self._running = True
        self._loop()
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                chunk = self._ser.read(512)
            except serial.SerialException as e:
                self.error.emit(str(e))
                break
            except OSError:
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
                    if len(self._scg_accum)  > 2048: self._scg_accum  = self._scg_accum[-2048:]
                    if len(self._ppg_accum)  > 1024: self._ppg_accum  = self._ppg_accum[-1024:]
                    if len(self._beat_accum) > 256:  self._beat_accum = self._beat_accum[-256:]

    def drain(self):
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
        self.setPriority(QThread.HighPriority)

    def run(self):
        self._reader.start()

    def stop(self):
        self._reader.stop()
        self.quit()
        if not self.wait(500):
            self.terminate()
            self.wait(500)


# ═══════════════════════════════════════════════════════════════════════════════
# Async CSV writer
# ═══════════════════════════════════════════════════════════════════════════════

_WRITER_STOP = object()

class AsyncCSVWriter:
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
            pass

    def close(self):
        self._q.put(_WRITER_STOP)
        self._thread.join(timeout=10)

    def _worker(self):
        with open(self._path, 'w', newline='', encoding='utf-8',
                  buffering=65536) as f:
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
# Ring buffer
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
# Styling & Real-time Constants
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_SECS     = 5
SAMPLE_RATE     = 256
PPG_SAMPLE_RATE = 100
WINDOW_N        = WINDOW_SECS * SAMPLE_RATE
WINDOW_PPG_N    = WINDOW_SECS * PPG_SAMPLE_RATE
BPM_HISTORY     = 8
MAX_PLOT_POINTS = 1280

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
    pw.showGrid(x=False, y=True, alpha=0.15)
    pw.getAxis('left').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('bottom').setTextPen(pg.mkPen(TEXT_DIM))
    pw.getAxis('left').setPen(pg.mkPen(BORDER))
    pw.getAxis('bottom').setPen(pg.mkPen(BORDER))
    pw.setClipToView(True)
    pw.setDownsampling(auto=True, mode="peak")
    pw.setMenuEnabled(False)
    pw.setMouseEnabled(x=False, y=True)
    pw.setTitle(f'<span style="color:{TEXT_DIM};font-size:9px;letter-spacing:2px;font-family:{FONT_UI};font-weight:bold;">{title}</span>')
    return pw


def raw_to_g(raw_int16: int) -> float:
    adc = float(raw_int16 & 0xFFFF)
    return (adc - ADC_ZERO_G) / ADC_COUNTS_PER_G


def ppg_norm(counts: float) -> float:
    return counts / PPG_MAX_COUNTS


REQUIRED_COLS = ["timestamp_ms", "x_g", "y_g", "z_g", "beat_event"]
OPTIONAL_PPG_COL = "ppg_raw"
DEFAULT_SEARCH_DIR = "SUBJECT_Data"
META_SUFFIX = "_meta.json"
ML_EXPORT_DIR = "Subject_Data_Segmented"
ML_TARGET_FS = 256


class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=3, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)

    def clear_and_get_axes(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        return ax


class MplPlotWidget(QWidget):
    def __init__(self, title: str = "", width: float = 8.0, height: float = 3.0, dpi: int = 100, sharex=None):
        super().__init__()
        self._title = title
        self._height = height
        self._dpi = dpi
        self.canvas = MplCanvas(width=width, height=height, dpi=dpi)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._axes = self.canvas.figure.add_subplot(111, sharex=sharex)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(int(height * dpi) + 110)

        if title:
            self._axes.set_title(title)

    @property
    def axes(self):
        return self._axes

    def clear(self):
        self._axes.cla()
        if self._title:
            self._axes.set_title(self._title)

    def draw(self):
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()


def _read_csv_from_source(uploaded_file, selected_path: str) -> pd.DataFrame:
    read_kwargs = {"comment": "#", "engine": "python"}
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, **read_kwargs)
    return pd.read_csv(selected_path, **read_kwargs)


def _resolve_meta_path(csv_path: str) -> Path:
    csv_file = Path(csv_path)
    return csv_file.with_name(f"{csv_file.stem}{META_SUFFIX}")


def _read_metadata_for_csv(csv_path: str) -> tuple[Optional[dict], str]:
    if not csv_path:
        return None, ""
    meta_path = _resolve_meta_path(csv_path)
    if not meta_path.exists():
        return None, str(meta_path)
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), str(meta_path)


def _get_csv_duration(path: Path) -> float:
    try:
        ts = pd.read_csv(path, usecols=["timestamp_ms"], comment="#")["timestamp_ms"]
        if len(ts) >= 2:
            return float((ts.iloc[-1] - ts.iloc[0]) / 1000.0)
    except Exception:
        pass
    return 0.0


def _generate_metadata_html(meta: Optional[dict], duration_s: float) -> str:
    if not meta:
        return f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8f9fa; padding: 15px; color: #718096;">
            <div style="font-size: 14px; font-weight: bold; color: #e53e3e; margin-bottom: 8px;">No Metadata File Found</div>
            <p>To view detailed metadata, please ensure a matching <code>_meta.json</code> file exists alongside your CSV.</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px; background: white; border: 1px solid #e2e8f0; border-radius: 6px;">
                <tr style="background-color: #f7fafc;">
                    <td style="padding: 10px; font-weight: bold; border-bottom: 1px solid #e2e8f0;">Session Metric</td>
                    <td style="padding: 10px; font-weight: bold; text-align: right; border-bottom: 1px solid #e2e8f0;">Value</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #edf2f7; color: #4a5568;">Recording Duration</td>
                    <td style="padding: 10px; text-align: right; font-family: monospace; border-bottom: 1px solid #edf2f7; font-weight: bold; color: #2d3748;">{duration_s:.3f} seconds</td>
                </tr>
            </table>
        </body>
        </html>
        """

    initials = meta.get("patient_initials", "N/A")
    age = meta.get("age", "N/A")
    sex = meta.get("sex", "N/A")
    weight = meta.get("weight_kg", None)
    height = meta.get("height_cm", None)
    bmi = meta.get("bmi", None)
    
    weight_str = f"{weight:.1f} kg" if isinstance(weight, (int, float)) else "N/A"
    height_str = f"{height:.1f} cm" if isinstance(height, (int, float)) else "N/A"
    bmi_str = f"{bmi:.1f}" if isinstance(bmi, (int, float)) else "N/A"

    session_start_raw = meta.get("session_start", "")
    session_start = session_start_raw.replace("T", " ") if session_start_raw else "N/A"
    if "." in session_start:
        session_start = session_start.split(".")[0]

    conditions = meta.get("cardiac_conditions", [])
    cond_badges = ""
    for cond in (conditions if isinstance(conditions, list) else [str(conditions)]):
        style = "background-color: #c6f6d5; color: #22543d;" if cond.lower() == "normal" else "background-color: #fed7d7; color: #9b2c2c;"
        cond_badges += f'<span style="border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold; margin-right: 4px; {style}">{cond}</span>'

    notes = meta.get("notes", "")

    scg_hz = meta.get("sample_rate_scg_hz", "N/A")
    ppg_hz = meta.get("sample_rate_ppg_hz", "N/A")
    
    filter_badge = f'<span style="border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold; background-color: #c6f6d5; color: #22543d;">Yes</span>' if meta.get("filter_enabled", False) else '<span style="border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold; background-color: #edf2f7; color: #4a5568;">No</span>'
    notch_badge = f'<span style="border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold; background-color: #c6f6d5; color: #22543d;">Yes</span>' if meta.get("notch_50hz_enabled", False) else '<span style="border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold; background-color: #edf2f7; color: #4a5568;">No</span>'

    html = f"""
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8f9fa; color: #2d3748; padding: 10px; margin: 0;">
        <table style="width: 100%; border-collapse: separate; border-spacing: 12px;">
            <tr>
                <!-- Patient Card -->
                <td style="width: 33%; vertical-align: top; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <div style="font-size: 15px; font-weight: bold; color: #2b6cb0; border-bottom: 2px solid #3182ce; padding-bottom: 6px; margin-bottom: 10px;">Patient Demographics</div>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Initials</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; font-weight: bold; color: #1a202c;">{initials}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Age</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; color: #2d3748;">{age} years</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Sex</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; color: #2d3748;">{sex}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Height</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; color: #2d3748;">{height_str}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Weight</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; color: #2d3748;">{weight_str}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; font-weight: 600; color: #4a5568;">BMI</td>
                            <td style="padding: 6px 0; text-align: right; font-weight: bold; color: #2b6cb0;">{bmi_str}</td>
                        </tr>
                    </table>
                </td>

                <!-- Session / Recording Card -->
                <td style="width: 34%; vertical-align: top; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <div style="font-size: 15px; font-weight: bold; color: #2b6cb0; border-bottom: 2px solid #3182ce; padding-bottom: 6px; margin-bottom: 10px;">Session Information</div>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Start Time</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; font-size: 12px; color: #2d3748;">{session_start}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Duration</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; font-weight: bold; color: #dd6b20;">{duration_s:.3f} s</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Cardiac Status</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right;">{cond_badges}</td>
                        </tr>
                    </table>
                    {f'<div style="margin-top: 10px; padding: 8px 12px; background-color: #ebf8ff; border-left: 4px solid #3182ce; border-radius: 4px; font-size: 12px; color: #2b6cb0; line-height: 1.4;"><strong>Notes:</strong> {notes}</div>' if notes else ''}
                </td>

                <!-- Hardware & Filtering Card -->
                <td style="width: 33%; vertical-align: top; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <div style="font-size: 15px; font-weight: bold; color: #2b6cb0; border-bottom: 2px solid #3182ce; padding-bottom: 6px; margin-bottom: 10px;">Device & Filtering</div>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">SCG rate</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; color: #2d3748; font-weight: bold;">{scg_hz} Hz</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">PPG rate</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right; color: #2d3748; font-weight: bold;">{ppg_hz} Hz</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; font-weight: 600; color: #4a5568;">Bandpass Filter</td>
                            <td style="padding: 6px 0; border-bottom: 1px solid #edf2f7; text-align: right;">{filter_badge}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; font-weight: 600; color: #4a5568;">50Hz Notch Filter</td>
                            <td style="padding: 6px 0; text-align: right;">{notch_badge}</td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    return html


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
    out["x_g"] = pd.to_numeric(out["x_g"], errors="coerce")
    out["y_g"] = pd.to_numeric(out["y_g"], errors="coerce")
    out["z_g"] = pd.to_numeric(out["z_g"], errors="coerce")
    out["beat_event"] = pd.to_numeric(out["beat_event"], errors="coerce").fillna(0).astype(int)
    if OPTIONAL_PPG_COL in out.columns:
        out[OPTIONAL_PPG_COL] = pd.to_numeric(out[OPTIONAL_PPG_COL], errors="coerce")

    out = out.dropna(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    return out


def _compute_rate(scg_df: pd.DataFrame) -> tuple[float, float]:
    if len(scg_df) < 2:
        return 0.0, 0.0
    duration_s = (scg_df["timestamp_ms"].iloc[-1] - scg_df["timestamp_ms"].iloc[0]) / 1000.0
    if duration_s <= 0:
        return 0.0, 0.0
    actual_hz = (len(scg_df) - 1) / duration_s
    return duration_s, actual_hz


def _compute_rate_from_ts(ts_ms: np.ndarray) -> tuple[float, float]:
    if len(ts_ms) < 2:
        return 0.0, 0.0
    duration_s = (float(ts_ms[-1]) - float(ts_ms[0])) / 1000.0
    if duration_s <= 0:
        return 0.0, 0.0
    actual_hz = (len(ts_ms) - 1) / duration_s
    return duration_s, actual_hz


def _bandpass_ppg(ppg_signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    if fs <= 0:
        return ppg_signal
    nyq = 0.5 * fs
    low = max(0.01, low_hz) / nyq
    high = min(0.99 * nyq, high_hz) / nyq
    if low >= high:
        return ppg_signal
    sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, ppg_signal)


def _detect_ppg_peaks(ppg_signal: np.ndarray, fs: float, max_bpm: float, prominence_factor: float) -> np.ndarray:
    if len(ppg_signal) < 3 or fs <= 0:
        return np.array([], dtype=int)
    min_distance = int(max(1, (60.0 / max(max_bpm, 1.0)) * fs))
    sig_range = np.percentile(ppg_signal, 95) - np.percentile(ppg_signal, 5)
    dynamic_prom = sig_range * prominence_factor if sig_range > 0 else prominence_factor
    peaks, _ = signal.find_peaks(ppg_signal, distance=min_distance, prominence=dynamic_prom)
    return peaks.astype(int)


def format_timestamp(seconds_val: float) -> str:
    hours = int(seconds_val // 3600)
    minutes = int((seconds_val % 3600) // 60)
    seconds = seconds_val % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:07.4f}"


def parse_timestamp_to_seconds(time_str: str) -> float:
    parts = str(time_str).strip().split(":")
    if len(parts) != 3:
        return 0.0
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def _build_peaks_payload(peak_seconds, record_name: str, peak_label: str) -> dict:
    timestamps = [format_timestamp(float(sec)) for sec in peak_seconds]
    return {f"{record_name}_{peak_label}_Peaks": timestamps}


def save_peaks_to_json(peaks_indices, fs, record_name, output_dir="Saved_Peaks", peak_label="AO") -> str:
    os.makedirs(output_dir, exist_ok=True)
    fs_safe = float(fs) if fs else 0.0
    peak_seconds = np.asarray(peaks_indices, dtype=float) / fs_safe if fs_safe > 0 else []
    data = _build_peaks_payload(peak_seconds, record_name, peak_label)
    out_path = os.path.join(output_dir, f"{record_name}_{peak_label}_Peaks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return out_path


def save_peaks_seconds_to_json(peak_seconds, record_name, output_dir="Saved_Peaks", peak_label="AO") -> str:
    os.makedirs(output_dir, exist_ok=True)
    data = _build_peaks_payload(peak_seconds, record_name, peak_label)
    out_path = os.path.join(output_dir, f"{record_name}_{peak_label}_Peaks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return out_path


def export_subject_for_ml(
    df,
    meta_fields,
    ao_peaks_indices,
    fs_proc,
    output_base_dir,
    ppg_peak_times_s=None,
    ppg_raw=None,
    ppg_fs=None,
    ppg_key="PPG_Peaks",
):
    if df is None or df.empty:
        raise ValueError("No SCG data available for export.")

    patient_id = meta_fields.get("patient_id") or "subject"
    target_fs = int(meta_fields.get("target_fs", ML_TARGET_FS))

    if fs_proc and float(fs_proc) > 0:
        original_fs = float(fs_proc)
    else:
        _, inferred_fs = _compute_rate_from_ts(df["timestamp_ms"].to_numpy(dtype=float))
        original_fs = float(inferred_fs) if inferred_fs > 0 else float(target_fs)

    scg_x = np.nan_to_num(df["x_g"].to_numpy(dtype=float))
    scg_y = np.nan_to_num(df["y_g"].to_numpy(dtype=float))
    scg_z = np.nan_to_num(df["z_g"].to_numpy(dtype=float))

    if original_fs != target_fs:
        new_len = int(round(len(scg_x) * target_fs / original_fs))
        scg_x = signal.resample(scg_x, new_len)
        scg_y = signal.resample(scg_y, new_len)
        scg_z = signal.resample(scg_z, new_len)
    else:
        new_len = len(scg_x)

    start_ts_ms = float(df["timestamp_ms"].iloc[0])
    ts_ms = start_ts_ms + np.arange(new_len, dtype=float) * (1000.0 / float(target_fs))
    host_ts_ms = ts_ms.copy()

    beat_event = np.zeros(new_len, dtype=int)
    if ppg_peak_times_s is not None and len(ppg_peak_times_s) > 0:
        peak_idx = np.round(np.asarray(ppg_peak_times_s, dtype=float) * float(target_fs)).astype(int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < new_len)]
        beat_event[peak_idx] = 1

    ppg_raw_series = None
    if ppg_raw is not None and len(ppg_raw) > 0 and ppg_fs and float(ppg_fs) > 0:
        if float(ppg_fs) != float(target_fs):
            ppg_raw_series = signal.resample(np.asarray(ppg_raw, dtype=float), new_len)
        else:
            ppg_raw_series = np.asarray(ppg_raw, dtype=float)

    export_df = pd.DataFrame({
        "timestamp_ms": ts_ms,
        "host_time_ms": host_ts_ms,
        "x_g": scg_x,
        "y_g": scg_y,
        "z_g": scg_z,
        "ppg_raw": ppg_raw_series if ppg_raw_series is not None else [""] * new_len,
        "beat_event": beat_event,
    })

    session_start = meta_fields.get("session_start")
    try:
        session_dt = datetime.fromisoformat(session_start) if session_start else datetime.now()
    except ValueError:
        session_dt = datetime.now()

    date_dir = Path(output_base_dir) / session_dt.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    csv_path = date_dir / f"{patient_id}.csv"
    meta_path = date_dir / f"{patient_id}_meta.json"

    conditions = meta_fields.get("cardiac_conditions") or ["Normal"]
    cond_str = ", ".join(conditions)

    meta_lines = [
        f"SCG/PPG Recording - {session_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"patient_initials,{meta_fields.get('patient_initials', '')}",
        f"age,{meta_fields.get('age', '')}",
        f"sex,{meta_fields.get('sex', '')}",
        f"weight_kg,{meta_fields.get('weight_kg', '')}",
        f"height_cm,{meta_fields.get('height_cm', '')}",
        f"bmi,{meta_fields.get('bmi', '')}",
        f"cardiac_conditions,{cond_str}",
        f"notes,{meta_fields.get('notes', '')}",
        f"sample_rate_scg_hz,{target_fs}",
        f"sample_rate_ppg_hz,{meta_fields.get('sample_rate_ppg_hz', PPG_SAMPLE_RATE)}",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        for line in meta_lines:
            handle.write(f"# {line}\n")
        export_df.to_csv(handle, index=False)

    meta_json = {
        "session_start": session_dt.isoformat(),
        "patient_initials": meta_fields.get("patient_initials", ""),
        "age": meta_fields.get("age"),
        "sex": meta_fields.get("sex"),
        "weight_kg": meta_fields.get("weight_kg"),
        "height_cm": meta_fields.get("height_cm"),
        "bmi": meta_fields.get("bmi"),
        "cardiac_conditions": conditions,
        "notes": meta_fields.get("notes", ""),
        "sample_rate_scg_hz": target_fs,
        "sample_rate_ppg_hz": meta_fields.get("sample_rate_ppg_hz", PPG_SAMPLE_RATE),
        "filter_enabled": meta_fields.get("filter_enabled"),
        "notch_50hz_enabled": meta_fields.get("notch_50hz_enabled"),
    }

    meta_json = {k: v for k, v in meta_json.items() if v not in (None, "")}
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_json, handle, indent=2)

    if ppg_peak_times_s is not None and len(ppg_peak_times_s) > 0:
        ppg_times = [format_timestamp(float(sec)) for sec in ppg_peak_times_s]
        ppg_path = date_dir / f"{patient_id}-PPG.json"
        with ppg_path.open("w", encoding="utf-8") as handle:
            json.dump({ppg_key: ppg_times}, handle, indent=2)

    if ao_peaks_indices is not None and len(ao_peaks_indices) > 0:
        save_peaks_to_json(ao_peaks_indices, fs_proc, patient_id, output_dir="Saved_Peaks", peak_label="AO")

    return {"csv_path": str(csv_path), "meta_path": str(meta_path)}


def _map_times_to_indices(time_axis_s: np.ndarray, event_times_s: np.ndarray) -> np.ndarray:
    if len(time_axis_s) == 0 or len(event_times_s) == 0:
        return np.array([], dtype=int)
    idx = np.searchsorted(time_axis_s, event_times_s)
    idx = np.clip(idx, 1, len(time_axis_s) - 1)
    prev_idx = idx - 1
    next_idx = idx
    prev_dt = np.abs(time_axis_s[prev_idx] - event_times_s)
    next_dt = np.abs(time_axis_s[next_idx] - event_times_s)
    chosen = np.where(next_dt < prev_dt, next_idx, prev_idx)
    return chosen.astype(int)


def resample_for_processing(raw_signal, fs_original, target_fs=500):
    if fs_original <= target_fs:
        return np.asarray(raw_signal), fs_original
    num_samples = int(len(raw_signal) * (target_fs / fs_original))
    return signal.resample(raw_signal, num_samples), target_fs


def apply_mti_filter(raw_signal):
    def mti_pass(sig, beta):
        return signal.lfilter([beta, -beta], [1, -beta], sig)

    x_beta1 = mti_pass(raw_signal, 0.9)
    x_beta2 = mti_pass(raw_signal, 0.99)
    y_filtered = x_beta2 - x_beta1
    y_detrended = signal.detrend(y_filtered)
    y_smoothed = signal.medfilt(y_detrended, kernel_size=5)
    return y_smoothed


def apply_scg_bandpass(raw_signal, fs, low_hz=1.0, high_hz=40.0):
    if fs <= 0:
        return raw_signal
    nyq = 0.5 * fs
    low = max(0.01, float(low_hz)) / nyq
    high = min(0.99 * nyq, float(high_hz)) / nyq
    if low >= high:
        return raw_signal
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, raw_signal)


def svmd(signal_in, max_alpha=2000, tau=0, tol=1e-6, stopc=3, init_omega=0):
    signal_in = np.array(signal_in).flatten()
    if len(signal_in) < 25:
        return np.empty((0, 0)), np.array([])
    if len(signal_in) % 2 > 0:
        signal_in = signal_in[:-1]
    if len(signal_in) < 25:
        return np.empty((0, 0)), np.array([])

    y = savgol_filter(signal_in, window_length=25, polyorder=8)
    signoise = signal_in - y

    save_T = len(signal_in)
    fs = 1.0

    T_half = save_T // 2
    f_mir = np.zeros(2 * save_T)
    f_mir_noise = np.zeros(2 * save_T)

    f_mir[0:T_half] = signal_in[T_half - 1 :: -1]
    f_mir_noise[0:T_half] = signoise[T_half - 1 :: -1]

    f_mir[T_half : save_T + T_half] = signal_in
    f_mir_noise[T_half : save_T + T_half] = signoise

    f_mir[save_T + T_half : 2 * save_T] = signal_in[save_T - 1 : T_half - 1 : -1]
    f_mir_noise[save_T + T_half : 2 * save_T] = signoise[save_T - 1 : T_half - 1 : -1]

    f = f_mir
    fnoise = f_mir_noise
    T = len(f)

    t = np.arange(1, T + 1) / T
    omega_freqs = t - 0.5 - 1.0 / T

    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_onesided = f_hat.copy()
    f_hat_onesided[0 : T // 2] = 0

    f_hat_n = np.fft.fftshift(np.fft.fft(fnoise))
    f_hat_n_onesided = f_hat_n.copy()
    f_hat_n_onesided[0 : T // 2] = 0

    noisepe = np.linalg.norm(f_hat_n_onesided, 2) ** 2

    N_max = 300
    omega_L = np.zeros(N_max)
    if init_omega == 0:
        omega_L[0] = 0.0
    else:
        omega_L[0] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1)))[0]

    minAlpha = 10
    Alpha = minAlpha

    n = 0
    m_val = 0
    SC2 = 0
    l = 0
    bf = 0

    BIC = []
    h_hat_Temp = []
    u_hat_Temp = []
    u_hat_i = []
    alpha_arr = []

    n2 = 0
    polm = []
    polm_temp = 1.0

    omega_d_Temp = []
    sigerror = []
    gamma = []
    normind = []

    lambda_arr = np.zeros((N_max, len(omega_freqs)), dtype=complex)
    u_hat_L = np.zeros((N_max, len(omega_freqs)), dtype=complex)

    while SC2 != 1:
        while Alpha < (max_alpha + 1):
            udiff = tol + np.finfo(float).eps
            while udiff > tol and n < N_max - 1:
                sum_h_hat = np.sum(h_hat_Temp, axis=0) if len(h_hat_Temp) > 0 else np.zeros(len(omega_freqs))
                sum_u_i = np.sum(u_hat_i, axis=0) if len(u_hat_i) > 0 else np.zeros(len(omega_freqs), dtype=complex)

                freq_diff = omega_freqs - omega_L[n]
                freq_diff_sq = freq_diff**2
                freq_diff_pow4 = freq_diff_sq**2
                alpha_sq = Alpha**2

                num_u = f_hat_onesided + (alpha_sq * freq_diff_pow4) * u_hat_L[n, :] + lambda_arr[n, :] / 2.0
                den_u = 1 + alpha_sq * freq_diff_pow4 * (1 + 2 * Alpha * freq_diff_sq) + sum_h_hat
                u_hat_L[n + 1, :] = num_u / den_u

                pos = slice(T // 2, T)
                mag_sq = np.abs(u_hat_L[n + 1, pos]) ** 2
                den_omega = np.sum(mag_sq)
                if den_omega > 0:
                    omega_L[n + 1] = np.dot(omega_freqs[pos], mag_sq) / den_omega
                else:
                    omega_L[n + 1] = omega_L[n]

                term1 = alpha_sq * freq_diff_pow4
                inner_paren = f_hat_onesided - u_hat_L[n + 1, :] - sum_u_i + lambda_arr[n, :] / 2.0
                num_tau = term1 * inner_paren
                den_tau = 1 + term1
                u_r_updated = num_tau / den_tau + sum_u_i

                lambda_arr[n + 1, :] = lambda_arr[n, :] + tau * (f_hat_onesided - u_hat_L[n + 1, :] - u_r_updated)

                udiff = np.finfo(float).eps
                diff_u = u_hat_L[n + 1, :] - u_hat_L[n, :]
                num_udiff = np.sum(diff_u * np.conj(diff_u)).real / T
                den_udiff = np.sum(u_hat_L[n, :] * np.conj(u_hat_L[n, :])).real / T

                if den_udiff > 0:
                    udiff += np.abs(num_udiff / den_udiff)
                udiff = np.abs(udiff)

                n += 1

            if abs(m_val - np.log(max_alpha)) > 1:
                m_val += 1
            else:
                m_val += 0.05
                bf += 1

            if bf >= 2:
                Alpha += 1

            if Alpha <= (max_alpha - 1):
                if bf == 1:
                    Alpha = max_alpha - 1
                else:
                    Alpha = np.exp(m_val)

                omega_L[0] = omega_L[n]
                temp_ud = u_hat_L[n, :].copy()
                n = 0
                lambda_arr = np.zeros((N_max, len(omega_freqs)), dtype=complex)
                u_hat_L = np.zeros((N_max, len(omega_freqs)), dtype=complex)
                u_hat_L[0, :] = temp_ud

        valid_omegas = omega_L[omega_L > 0]
        if len(valid_omegas) > 0:
            idx = max(0, min(n - 1, len(valid_omegas) - 1))
            omega_d_Temp.append(valid_omegas[idx])
        else:
            omega_d_Temp.append(0.0)

        u_hat_Temp.append(u_hat_L[n, :].copy())
        alpha_arr.append(Alpha)

        Alpha = minAlpha
        bf = 0

        if init_omega > 0:
            ii = 0
            while ii < 1 and n2 < 300:
                rand_omega = np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand())
                checkp = np.abs(np.array(omega_d_Temp) - rand_omega)
                if len(checkp[checkp < 0.02]) <= 0:
                    ii = 1
                n2 += 1
            omega_L_start = rand_omega
        else:
            omega_L_start = 0.0

        lambda_arr = np.zeros((N_max, len(omega_freqs)), dtype=complex)
        gamma.append(1)

        h_hat_new = gamma[l] / ((alpha_arr[l] ** 2) * (omega_freqs - omega_d_Temp[l]) ** 4)
        h_hat_Temp.append(h_hat_new)
        u_hat_i.append(u_hat_Temp[l])

        sum_u_i = np.sum(u_hat_i, axis=0) if len(u_hat_i) > 0 else np.zeros(len(omega_freqs), dtype=complex)

        if stopc == 1:
            sigerror_val = np.linalg.norm(f_hat_onesided - sum_u_i, 2) ** 2
            sigerror.append(sigerror_val)
            if n2 >= 300 or sigerror[l] <= np.round(noisepe):
                SC2 = 1

        elif stopc == 2:
            sum_u_temp = np.sum(u_hat_Temp, axis=0)
            normind_val = (1 / T) * np.linalg.norm(sum_u_temp - f_hat_onesided, 2) ** 2 / ((1 / T) * np.linalg.norm(f_hat_onesided, 2) ** 2)
            normind.append(normind_val)
            if n2 >= 300 or normind[l] < 0.005:
                SC2 = 1

        elif stopc == 3:
            sigerror_val = np.linalg.norm(f_hat_onesided - sum_u_i, 2) ** 2
            sigerror.append(sigerror_val)
            BIC_val = 2 * T * np.log(sigerror[l]) + (3 * (l + 1)) * np.log(2 * T)
            BIC.append(BIC_val)
            if l > 0 and BIC[l] > BIC[l - 1]:
                SC2 = 1

        else:
            term = (4 * alpha_arr[l] * u_hat_i[l]) / (1 + 2 * alpha_arr[l] * (omega_freqs - omega_d_Temp[l]) ** 2)
            dot_prod = np.sum(term * np.conj(u_hat_i[l]))
            polm_val = np.abs(dot_prod)

            if l == 0:
                polm.append(polm_val)
                polm_temp = polm[0]
                polm[0] = 1.0
            else:
                polm.append(polm_val / polm_temp)

            if l > 0 and abs(polm[l] - polm[l - 1]) < 0.001:
                SC2 = 1

        u_hat_L = np.zeros((N_max, len(omega_freqs)), dtype=complex)
        omega_L = np.zeros(N_max)
        omega_L[0] = omega_L_start
        n = 0
        l += 1
        m_val = 0
        n2 = 0

    omega_final = np.array(omega_d_Temp)
    L = len(omega_final)
    u_hat = np.zeros((T, L), dtype=complex)

    for idx in range(L):
        u_hat[T // 2 : T, idx] = u_hat_Temp[idx][T // 2 : T]
        u_hat[1 : T // 2, idx] = np.conj(u_hat_Temp[idx][T // 2 + 1 : T][::-1])
        u_hat[0, idx] = np.conj(u_hat[-1, idx])

    u = np.zeros((L, T))
    for idx in range(L):
        u[idx, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, idx])))

    indic = np.argsort(omega_final)
    omega_final = omega_final[indic]
    u = u[indic, :]
    u_final = u[:, T // 4 : 3 * T // 4]
    return u_final, omega_final


def select_ao_modes(modes, omegas=None, fs=None, freq_cutoff_hz=100):
    if len(modes) == 0:
        return np.array([]), np.array([]), np.nan, np.array([], dtype=int)
    wfs = []
    for mode in modes:
        rms = np.sqrt(np.mean(mode**2))
        mav = np.mean(np.abs(mode))
        wfs.append(rms / mav if mav > 0 else 0.0)
    wfs = np.array(wfs)

    if omegas is not None and fs is not None and len(omegas) == len(modes):
        center_freq_hz = np.abs(omegas) * fs
        eligible_mask = center_freq_hz <= freq_cutoff_hz
    else:
        eligible_mask = np.ones(len(wfs), dtype=bool)

    if np.any(eligible_mask):
        wf_mean = np.mean(wfs[eligible_mask])
        selected_indices = np.where((wfs > wf_mean) & eligible_mask)[0]
    else:
        wf_mean = np.nan
        selected_indices = np.array([], dtype=int)

    if len(selected_indices) > 0:
        s_ao = np.sum(modes[selected_indices], axis=0)
    else:
        s_ao = np.zeros_like(modes[0])

    return s_ao, wfs, wf_mean, selected_indices


def extract_ao_peaks(s_ao, fs, prominence_factor=0.05, power=7):
    taper_window = signal.windows.tukey(len(s_ao), alpha=0.05)
    s_ao_tapered = s_ao * taper_window
    s_ao_7 = np.sign(s_ao_tapered) * np.abs(s_ao_tapered) ** power
    analytic_signal = signal.hilbert(s_ao_7)
    envelope = np.abs(analytic_signal)
    window_width = max(1, int(fs / 10))
    smoothed_env = np.convolve(envelope, np.ones(window_width) / window_width, mode="same")
    min_distance = int(0.4 * fs)
    prom_threshold = max(np.percentile(smoothed_env, 90) * prominence_factor, np.percentile(smoothed_env, 10))
    peaks, _ = signal.find_peaks(smoothed_env, distance=min_distance, prominence=prom_threshold)
    return s_ao_7, envelope, smoothed_env, peaks


def sqa_kurtosis(segment, threshold=7.0):
    return kurtosis(segment) > threshold


def sqa_zcr(segment, fs, low_hz=0.5, high_hz=5.0):
    zcr = np.sum(np.diff(np.sign(segment)) != 0) / (len(segment) / fs)
    return zcr < low_hz or zcr > high_hz


def sqa_flatline(segment, cv_thresh=0.01, diff_thresh=1e-4):
    cv = np.std(segment) / (np.abs(np.mean(segment)) + 1e-9)
    return cv < cv_thresh or np.max(np.abs(np.diff(segment))) < diff_thresh


def sqa_envelope(segment, threshold=2.5):
    env = np.abs(hilbert(segment))
    return (np.std(env) / (np.mean(env) + 1e-9)) > threshold


def _derive_rms_bounds(seg_rms, rms_low_percentile=20, rms_high_percentile=80, rms_low_mad_mult=2.0, rms_high_mad_mult=4.0):
    valid_rms = np.asarray(seg_rms, dtype=float)
    valid_rms = valid_rms[np.isfinite(valid_rms)]
    if len(valid_rms) == 0:
        return 1e-10, float("inf")

    rms_low_perc = float(np.percentile(valid_rms, rms_low_percentile))
    rms_high_perc = float(np.percentile(valid_rms, rms_high_percentile))
    rms_med = float(np.median(valid_rms))
    rms_mad = float(np.median(np.abs(valid_rms - rms_med)))

    rms_low_thr = max(rms_low_perc, rms_med - rms_low_mad_mult * rms_mad) if rms_mad > 1e-12 else rms_low_perc
    rms_high_thr = max(rms_high_perc, rms_med + rms_high_mad_mult * rms_mad) if rms_mad > 1e-12 else rms_high_perc * 3.0
    return float(rms_low_thr), float(rms_high_thr)


def sqa_rms(segment, rms_low_thr, rms_high_thr):
    seg_det = signal.detrend(np.asarray(segment, dtype=float))
    rms_value = float(np.sqrt(np.mean(seg_det ** 2)))
    return (rms_value < rms_low_thr) or (rms_value > rms_high_thr), rms_value


def sqa_combined(
    segment,
    fs,
    min_flags=2,
    kurt_thresh=7.0,
    zcr_low=0.5,
    zcr_high=5.0,
    cv_thresh=0.01,
    diff_thresh=1e-4,
    env_thresh=2.5,
    rms_low_thr=1e-10,
    rms_high_thr=float("inf"),
):
    rms_flag, _ = sqa_rms(segment, rms_low_thr, rms_high_thr)
    flags = [
        sqa_kurtosis(segment, kurt_thresh),
        sqa_zcr(segment, fs, zcr_low, zcr_high),
        sqa_flatline(segment, cv_thresh, diff_thresh),
        sqa_envelope(segment, env_thresh),
        rms_flag,
    ]
    return sum(flags) >= min_flags, flags


def combined_sqa_for_signal(
    signal_in,
    fs,
    segment_seconds=4.0,
    min_flags=2,
    kurt_thresh=7.0,
    zcr_low=0.5,
    zcr_high=5.0,
    cv_thresh=0.01,
    diff_thresh=1e-4,
    env_thresh=2.5,
    rms_low_percentile=20,
    rms_high_percentile=80,
    rms_low_mad_mult=2.0,
    rms_high_mad_mult=4.0,
):
    x = np.asarray(signal_in, dtype=float).flatten()
    n = len(x)
    if n == 0 or fs <= 0:
        return {
            "segment_starts": np.array([]),
            "segment_ends": np.array([]),
            "bad_mask": np.array([], dtype=bool),
            "flags": np.empty((0, 5), dtype=bool),
            "seg_rms": np.array([]),
            "rms_bounds": {"low": np.nan, "high": np.nan},
            "method": "combined",
        }

    seg_len = max(1, int(segment_seconds * fs))
    seg_starts = np.arange(0, n, seg_len)
    seg_ends = np.minimum(seg_starts + seg_len, n)

    seg_rms_all = []
    for s, e in zip(seg_starts, seg_ends):
        seg = x[s:e]
        if len(seg) < 16:
            seg_rms_all.append(np.nan)
            continue
        seg_rms_all.append(float(np.sqrt(np.mean(signal.detrend(seg) ** 2))))

    rms_low_thr, rms_high_thr = _derive_rms_bounds(
        seg_rms_all,
        rms_low_percentile=rms_low_percentile,
        rms_high_percentile=rms_high_percentile,
        rms_low_mad_mult=rms_low_mad_mult,
        rms_high_mad_mult=rms_high_mad_mult,
    )

    bad_mask = []
    flags_all = []
    for s, e, rms_value in zip(seg_starts, seg_ends, seg_rms_all):
        seg = x[s:e]
        if len(seg) < 16:
            bad_mask.append(True)
            flags_all.append([True, True, True, True, True])
            continue
        is_bad, flags = sqa_combined(
            seg,
            fs,
            min_flags=min_flags,
            kurt_thresh=kurt_thresh,
            zcr_low=zcr_low,
            zcr_high=zcr_high,
            cv_thresh=cv_thresh,
            diff_thresh=diff_thresh,
            env_thresh=env_thresh,
            rms_low_thr=rms_low_thr,
            rms_high_thr=rms_high_thr,
        )
        bad_mask.append(is_bad)
        flags_all.append(flags)

    return {
        "segment_starts": seg_starts,
        "segment_ends": seg_ends,
        "bad_mask": np.asarray(bad_mask, dtype=bool),
        "flags": np.asarray(flags_all, dtype=bool),
        "seg_rms": np.asarray(seg_rms_all, dtype=float),
        "rms_bounds": {"low": float(rms_low_thr), "high": float(rms_high_thr)},
        "method": "combined",
    }


def build_sample_bad_mask(signal_len, sqa_result):
    sample_bad = np.zeros(signal_len, dtype=bool)
    seg_starts = sqa_result.get("segment_starts", [])
    seg_ends = sqa_result.get("segment_ends", [])
    bad_mask = sqa_result.get("bad_mask", [])
    for s, e, is_bad in zip(seg_starts, seg_ends, bad_mask):
        sample_bad[int(s) : int(e)] = bool(is_bad)
    return sample_bad


def _iqr_inlier_mask(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=bool)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return np.ones(values.shape, dtype=bool)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (values >= lower) & (values <= upper)


def _estimate_peak_lag_samples(detected_peaks, reference_peaks, fs_ao, fs_ref):
    detected = np.asarray(detected_peaks, dtype=float)
    reference = np.asarray(reference_peaks, dtype=float)
    if detected.size == 0 or reference.size == 0:
        return 0.0

    detected_s = detected / float(fs_ao)
    reference_s = reference / float(fs_ref)
    min_lag_s = 0.05
    max_lag_s = 0.50
    if max_lag_s <= min_lag_s:
        return 0.0

    diffs = []
    for ref in reference_s:
        lo = ref - max_lag_s
        hi = ref - min_lag_s
        nearby = detected_s[(detected_s >= lo) & (detected_s <= hi)]
        if nearby.size:
            diffs.extend(nearby - ref)

    if len(diffs) == 0:
        return 0.0

    diffs = np.asarray(diffs, dtype=float)
    bin_width = 0.001
    diffs_int = np.round(diffs / bin_width).astype(int)
    offset = -int(diffs_int.min())
    counts = np.bincount(diffs_int + offset)
    best_lag = (int(np.argmax(counts)) - offset) * bin_width
    return float(best_lag)


def _ptt_window_samples(lag_seconds, tolerance_seconds=0.15):
    window_min = lag_seconds - float(tolerance_seconds)
    window_max = lag_seconds + float(tolerance_seconds)
    if window_min >= window_max:
        window_min = lag_seconds - float(tolerance_seconds)
        window_max = lag_seconds + float(tolerance_seconds)
    return window_min, window_max


def _match_peaks_by_lag(detected_peaks, reference_peaks, fs_ao, fs_ref, tolerance_seconds=0.15):
    detected = np.asarray(detected_peaks, dtype=float)
    reference = np.asarray(reference_peaks, dtype=float)
    if detected.size == 0 or reference.size == 0:
        return [], []

    detected = np.sort(detected)
    reference = np.sort(reference)
    detected_s = detected / float(fs_ao)
    reference_s = reference / float(fs_ref)
    lag_seconds = _estimate_peak_lag_samples(detected, reference, fs_ao, fs_ref)
    window_min, window_max = _ptt_window_samples(lag_seconds, tolerance_seconds)

    used = np.zeros(len(detected_s), dtype=bool)
    matched_detected = []
    matched_reference = []

    for ref_idx, ref in enumerate(reference_s):
        candidates = np.where((~used) & (detected_s - ref >= window_min) & (detected_s - ref <= window_max))[0]
        if len(candidates) == 0:
            continue
        closest_idx = candidates[np.argmin(np.abs(detected_s[candidates] - ref))]
        used[closest_idx] = True
        matched_detected.append(int(closest_idx))
        matched_reference.append(int(ref_idx))
    return matched_detected, matched_reference


def _calculate_high_corr_ptt(ao_peaks, ppg_peaks, fs_ao, fs_ref, window_secs=15.0):
    """Scan sliding windows to find the segment with the highest AO-AO vs PPG-PPG
    interval correlation. Both peak arrays are converted to seconds internally.
    Returns (ptt_seconds, best_segment_info) where best_segment_info has keys
    't_start', 't_end', 'corr' in absolute time, or None if no valid window found."""
    ao_peaks = np.asarray(ao_peaks, dtype=float)
    ppg_peaks = np.asarray(ppg_peaks, dtype=float)
    if ao_peaks.size < 2 or ppg_peaks.size < 2:
        return 0.0, None

    ao_s = np.sort(ao_peaks) / float(fs_ao)
    ppg_s = np.sort(ppg_peaks) / float(fs_ref)

    # Global PTT fallback: exclusive nearest-preceding-AO-peak lag across all data
    global_lags = []
    used_ao_g = np.zeros(len(ao_s), dtype=bool)
    for p in ppg_s:
        diffs = p - ao_s
        valid = np.where((diffs >= 0.05) & (diffs <= 0.60) & ~used_ao_g)[0]
        if len(valid) > 0:
            chosen = valid[np.argmin(diffs[valid])]
            global_lags.append(diffs[chosen])
            used_ao_g[chosen] = True
    global_ptt = float(np.median(global_lags)) if len(global_lags) >= 2 else 0.2

    total_duration = max(ao_s[-1], ppg_s[-1]) if ao_s.size > 0 and ppg_s.size > 0 else 0.0
    data_start = min(ao_s[0], ppg_s[0])        # start windows at first available data
    step_secs = 2.0

    best_corr = -1.0
    best_ptt = global_ptt
    best_t_start = None
    best_t_end = None

    t_start = data_start
    while t_start + window_secs <= total_duration:
        t_end = t_start + window_secs

        win_ao = ao_s[(ao_s >= t_start) & (ao_s < t_end)]
        win_ppg = ppg_s[(ppg_s >= t_start) & (ppg_s < t_end)]

        if len(win_ao) >= 6 and len(win_ppg) >= 6:
            # Beat-to-beat intervals (ms) and their midpoint times (s)
            ao_ivals = np.diff(win_ao) * 1000.0
            ppg_ivals = np.diff(win_ppg) * 1000.0
            ao_centers = (win_ao[:-1] + win_ao[1:]) / 2.0
            ppg_centers = (win_ppg[:-1] + win_ppg[1:]) / 2.0

            # Match each PPG interval to the nearest unmatched AO interval by center time.
            # Since both signals have the same heart rate, centres should be within
            # half an RR interval of each other.
            median_rr = float(np.median(ppg_ivals)) / 1000.0
            max_dt = min(0.55 * median_rr, 0.6) if median_rr > 0 else 0.6

            matched_ao_iv = []
            matched_ppg_iv = []
            used_ao = np.zeros(len(ao_centers), dtype=bool)
            for pi in range(len(ppg_centers)):
                dists = np.abs(ao_centers - ppg_centers[pi])
                dists[used_ao] = np.inf
                best_i = int(np.argmin(dists))
                if dists[best_i] <= max_dt:
                    matched_ao_iv.append(ao_ivals[best_i])
                    matched_ppg_iv.append(ppg_ivals[pi])
                    used_ao[best_i] = True

            if len(matched_ao_iv) >= 5:
                mao = np.array(matched_ao_iv)
                mppg = np.array(matched_ppg_iv)
                # Guard: skip windows where variance is too low (would give trivial r=1)
                if np.std(mao) > 1.5 and np.std(mppg) > 1.5:
                    corr = np.corrcoef(mao, mppg)[0, 1]
                    if np.isfinite(corr) and corr > best_corr:
                        best_corr = corr
                        # PTT: exclusive nearest-preceding-AO-peak lag, median
                        lags = []
                        used_ao_pk = np.zeros(len(win_ao), dtype=bool)
                        for p in win_ppg:
                            diffs = p - win_ao
                            valid = np.where((diffs >= 0.05) & (diffs <= 0.60) & ~used_ao_pk)[0]
                            if len(valid) > 0:
                                chosen = valid[np.argmin(diffs[valid])]
                                lags.append(diffs[chosen])
                                used_ao_pk[chosen] = True
                        if len(lags) >= 2:
                            best_ptt = float(np.median(lags))
                        best_t_start = t_start
                        best_t_end = t_end

        t_start += step_secs

    if best_corr >= 0.60 and best_t_start is not None:
        segment_info = {"t_start": best_t_start, "t_end": best_t_end, "corr": best_corr}
        return best_ptt, segment_info

    # Fall back to global PTT — no best segment to highlight
    return global_ptt, None


def compute_ptt_metrics(ao_peaks, ppg_peaks, fs_ao, fs_ref, tolerance_seconds=0.15):
    ao_peaks = np.asarray(ao_peaks, dtype=float)
    ppg_peaks = np.asarray(ppg_peaks, dtype=float)
    if ao_peaks.size == 0 or ppg_peaks.size == 0:
        return None

    ao_sorted = np.sort(ao_peaks)
    ppg_sorted = np.sort(ppg_peaks)
    ao_s = ao_sorted / float(fs_ao)
    ppg_s = ppg_sorted / float(fs_ref)
    matched_ao_idx, matched_ppg_idx = _match_peaks_by_lag(ao_sorted, ppg_sorted, fs_ao, fs_ref, tolerance_seconds)
    if len(matched_ao_idx) == 0:
        return None

    ptt_ms = (ppg_s[np.asarray(matched_ppg_idx)] - ao_s[np.asarray(matched_ao_idx)]) * 1000.0
    ao_intervals_ms = np.diff(ao_s) * 1000.0
    ao_interval_series = []
    ptt_series = []
    for ao_idx, ptt_value in zip(matched_ao_idx, ptt_ms):
        if ao_idx == 0:
            continue
        ao_interval_series.append(ao_intervals_ms[ao_idx - 1])
        ptt_series.append(ptt_value)

    ptt_corr = np.corrcoef(ptt_series, ao_interval_series)[0, 1] if len(ptt_series) > 1 else np.nan
    return {"mean_ptt_ms": float(np.mean(ptt_ms)), "std_ptt_ms": float(np.std(ptt_ms)), "ptt_rr_correlation": ptt_corr}


def match_intervals_by_time(ao_peaks, ref_peaks, fs, apply_iqr=False):
    ao_peaks = np.asarray(ao_peaks, dtype=float)
    ref_peaks = np.asarray(ref_peaks, dtype=float)

    if len(ao_peaks) < 2 or len(ref_peaks) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    ao_intervals_ms = np.diff(ao_peaks) / fs * 1000.0
    ref_intervals_ms = np.diff(ref_peaks) / fs * 1000.0
    ao_centers_s = (ao_peaks[:-1] + ao_peaks[1:]) / 2.0 / fs
    ref_centers_s = (ref_peaks[:-1] + ref_peaks[1:]) / 2.0 / fs

    if len(ao_centers_s) == 0 or len(ref_centers_s) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    median_ref_seconds = np.median(ref_intervals_ms) / 1000.0
    if not np.isfinite(median_ref_seconds) or median_ref_seconds <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    max_time_diff = 0.5 * median_ref_seconds
    matched_ao = []
    matched_ref = []
    matched_ao_centers = []
    matched_ref_centers = []

    for idx, ao_center in enumerate(ao_centers_s):
        nearest_idx = int(np.argmin(np.abs(ref_centers_s - ao_center)))
        time_diff = abs(ref_centers_s[nearest_idx] - ao_center)
        if time_diff < max_time_diff:
            matched_ao.append(ao_intervals_ms[idx])
            matched_ref.append(ref_intervals_ms[nearest_idx])
            matched_ao_centers.append(ao_center)
            matched_ref_centers.append(ref_centers_s[nearest_idx])

    matched_ao = np.asarray(matched_ao, dtype=float)
    matched_ref = np.asarray(matched_ref, dtype=float)
    matched_ao_centers = np.asarray(matched_ao_centers, dtype=float)
    matched_ref_centers = np.asarray(matched_ref_centers, dtype=float)
    if matched_ao.size == 0 or matched_ref.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    removed_ao_centers = np.array([])
    removed_ref_centers = np.array([])
    if apply_iqr:
        ao_mask = _iqr_inlier_mask(matched_ao)
        ref_mask = _iqr_inlier_mask(matched_ref)
        keep_mask = ao_mask & ref_mask
        removed_ao_centers = matched_ao_centers[~keep_mask]
        removed_ref_centers = matched_ref_centers[~keep_mask]
    else:
        keep_mask = np.ones(matched_ao.shape, dtype=bool)

    return (
        matched_ao[keep_mask],
        matched_ref[keep_mask],
        matched_ao_centers[keep_mask],
        matched_ref_centers[keep_mask],
        removed_ao_centers,
        removed_ref_centers,
    )


def compute_paper_metrics(ao_intervals_ms, ref_intervals_ms):
    ao_intervals_ms = np.asarray(ao_intervals_ms, dtype=float)
    ref_intervals_ms = np.asarray(ref_intervals_ms, dtype=float)
    if len(ao_intervals_ms) == 0 or len(ref_intervals_ms) == 0:
        return None

    ao_hr = 60000.0 / ao_intervals_ms
    ref_hr = 60000.0 / ref_intervals_ms

    mean_scg_hr = float(np.mean(ao_hr))
    mean_ref_hr = float(np.mean(ref_hr))
    are = abs(mean_scg_hr - mean_ref_hr) / mean_ref_hr if mean_ref_hr != 0 else np.nan
    aae = float(np.mean(np.abs(ao_hr - ref_hr)))
    aaep = float(np.mean(np.abs(ao_hr - ref_hr) / ref_hr) * 100.0)

    diff_intervals = ao_intervals_ms - ref_intervals_ms
    bias = float(np.mean(diff_intervals))
    std_diff = float(np.std(diff_intervals))
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff

    return {
        "ARE": are,
        "AAE": aae,
        "AAEP": aaep,
        "mean_scg_hr": mean_scg_hr,
        "mean_ref_hr": mean_ref_hr,
        "ba_bias": bias,
        "ba_upper_loa": loa_upper,
        "ba_lower_loa": loa_lower,
    }


def compute_detection_metrics(detected_peaks, reference_peaks, fs_ao, fs_ref, tolerance_seconds=0.15, already_aligned=False):
    detected = np.asarray(detected_peaks, dtype=float)
    reference = np.asarray(reference_peaks, dtype=float)
    if len(reference) == 0:
        return None

    detected = np.sort(detected)
    reference = np.sort(reference)
    detected_s = detected / float(fs_ao)
    reference_s = reference / float(fs_ref)
    if already_aligned:
        lag_seconds = 0.0
    else:
        lag_seconds = _estimate_peak_lag_samples(detected, reference, fs_ao, fs_ref)
    window_min, window_max = _ptt_window_samples(lag_seconds, tolerance_seconds)

    used = np.zeros(len(detected_s), dtype=bool)
    matched_ref = np.zeros(len(reference_s), dtype=bool)
    tp = 0

    for ref_idx, ref in enumerate(reference_s):
        candidates = np.where((~used) & (detected_s - ref >= window_min) & (detected_s - ref <= window_max))[0]
        if len(candidates) == 0:
            continue
        closest_idx = candidates[np.argmin(np.abs(detected_s[candidates] - ref))]
        used[closest_idx] = True
        matched_ref[ref_idx] = True
        tp += 1

    fp = int(np.sum(~used))
    fn = int(np.sum(~matched_ref))
    fp_peaks = detected[~used]
    fn_peaks = reference[~matched_ref]

    se = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else np.nan
    p = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else np.nan
    acc = tp / (tp + fp + fn) * 100.0 if (tp + fp + fn) > 0 else np.nan
    der = (fp + fn) / (tp + fn) * 100.0 if (tp + fn) > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SE": se,
        "P": p,
        "ACC": acc,
        "DER": der,
        "fp_peaks": fp_peaks,
        "fn_peaks": fn_peaks,
        "fp_times_s": fp_peaks / float(fs_ao),
        "fn_times_s": fn_peaks / float(fs_ref),
    }


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _table_from_dataframe(table: QTableWidget, df: pd.DataFrame):
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(0)
    if df is None or df.empty:
        return
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for row_index, row in enumerate(df.itertuples(index=False)):
        for col_index, value in enumerate(row):
            table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
    table.resizeColumnsToContents()


class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_result = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, mode: str, inputs: dict):
        super().__init__()
        self.mode = mode
        self.inputs = inputs

    def run(self):
        try:
            if self.mode == "window":
                result = run_window_analysis(self.inputs)
            else:
                result = run_full_record_analysis(self.inputs, progress_callback=self.progress.emit, status_callback=self.status.emit)
            self.finished_result.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


def _current_ppg_info(df: pd.DataFrame, beat_times_s: np.ndarray, beat_source: str, ppg_bp_low: float, ppg_bp_high: float, ppg_max_bpm: float, ppg_prom: float, fs_proc: float):
    if df.empty or OPTIONAL_PPG_COL not in df.columns:
        return {
            "ppg_vis_filtered": None,
            "ppg_vis_peaks_idx": np.array([], dtype=int),
            "ppg_vis_fs": 0.0,
            "ref_fs": 100.0,
            "ppg_peaks_ref": np.array([], dtype=int),
            "ppg_peaks_full": np.array([], dtype=int),
            "ppg_time_s": np.array([], dtype=float),
            "ppg_peak_times_s": np.array([], dtype=float),
            "beat_times_s": beat_times_s,
        }

    ppg_ts = df["timestamp_ms"].to_numpy(dtype=float)
    _, ppg_vis_fs = _compute_rate_from_ts(ppg_ts)
    ppg_signal = df[OPTIONAL_PPG_COL].to_numpy(dtype=float)
    ppg_vis_filtered = _bandpass_ppg(ppg_signal, ppg_vis_fs, float(ppg_bp_low), float(ppg_bp_high))
    ppg_vis_peaks_idx = _detect_ppg_peaks(ppg_vis_filtered, ppg_vis_fs, float(ppg_max_bpm), float(ppg_prom))
    ppg_time_s = (ppg_ts - float(ppg_ts[0])) / 1000.0 if len(ppg_ts) > 0 else np.array([], dtype=float)
    if beat_source == "Detect from PPG raw" and len(ppg_vis_peaks_idx) > 0:
        beat_times_s = ppg_time_s[ppg_vis_peaks_idx]

    ref_fs = float(ppg_vis_fs) if beat_source == "Detect from PPG raw" and ppg_vis_fs > 0 else 100.0
    ppg_peaks_ref = (beat_times_s * ref_fs).astype(int) if len(beat_times_s) > 0 else np.array([], dtype=int)
    ppg_peaks_full = (beat_times_s * float(fs_proc)).astype(int) if len(beat_times_s) > 0 else np.array([], dtype=int)
    ppg_peak_times_s = ppg_time_s[ppg_vis_peaks_idx] if len(ppg_vis_peaks_idx) > 0 else np.array([], dtype=float)

    return {
        "ppg_vis_filtered": ppg_vis_filtered,
        "ppg_vis_peaks_idx": ppg_vis_peaks_idx,
        "ppg_vis_fs": float(ppg_vis_fs),
        "ref_fs": float(ref_fs),
        "ppg_peaks_ref": ppg_peaks_ref,
        "ppg_peaks_full": ppg_peaks_full,
        "ppg_time_s": ppg_time_s,
        "ppg_peak_times_s": ppg_peak_times_s,
        "beat_times_s": beat_times_s,
    }


def _processing_signal(scg_raw: np.ndarray, fs_infer: float, target_fs: float, preprocessing_mode: str):
    scg_proc_full, fs_proc = resample_for_processing(scg_raw, fs_infer, target_fs=target_fs)
    if preprocessing_mode == "MTI + Detrend + Median":
        scg_proc_full = apply_mti_filter(scg_proc_full)
    elif preprocessing_mode == "Bandpass 1-40 Hz":
        scg_proc_full = apply_scg_bandpass(scg_proc_full, fs_proc, low_hz=1.0, high_hz=40.0)
    return scg_proc_full, fs_proc


def run_window_analysis(inputs: dict) -> dict:
    scg_proc_full = inputs["scg_proc_full"]
    fs_proc = float(inputs["fs_proc"])
    start_time = float(inputs["start_time"])
    window_size = float(inputs["window_size"])
    start_idx = int(start_time * fs_proc)
    end_idx = min(int((start_time + window_size) * fs_proc), len(scg_proc_full))
    scg_window = scg_proc_full[start_idx:end_idx]
    if len(scg_window) < 25:
        return {"warning": "Window too short for SVMD."}

    # Slice the raw signal window if raw signal is available
    if "scg_raw" in inputs and "fs_infer" in inputs and len(inputs["scg_raw"]) > 0:
        scg_raw = inputs["scg_raw"]
        fs_infer = float(inputs["fs_infer"])
        start_raw_idx = int(start_time * fs_infer)
        end_raw_idx = min(int((start_time + window_size) * fs_infer), len(scg_raw))
        scg_raw_window = scg_raw[start_raw_idx:end_raw_idx]
        if len(scg_raw_window) == len(scg_window):
            scg_raw_window_resampled = scg_raw_window
        else:
            scg_raw_window_resampled = signal.resample(scg_raw_window, len(scg_window)) if len(scg_raw_window) > 0 else np.array([])
    else:
        scg_raw_window_resampled = np.array([])

    modes, omegas = svmd(scg_window, max_alpha=int(inputs["svmd_alpha"]), tau=0, stopc=3)
    if len(omegas) == 0:
        return {"warning": "SVMD returned no modes. Try adjusting parameters."}

    s_ao, wfs, wf_mean, selected_idx = select_ao_modes(modes, omegas, fs_proc)
    s_ao_7, envelope, smoothed_env, peaks = extract_ao_peaks(s_ao, fs_proc, float(inputs["prominence_factor"]), power=int(inputs["power_exp"]))

    center_freq_hz = np.abs(omegas) * fs_proc

    ppg_peaks_full = inputs["ppg_peaks_full"]
    beat_times_s = inputs["beat_times_s"]
    ppg_peaks_ref = inputs["ppg_peaks_ref"]
    ppg_window_mask = (beat_times_s >= start_time) & (beat_times_s < start_time + window_size)
    ppg_peaks_window_ref = ppg_peaks_ref[ppg_window_mask] if len(ppg_peaks_ref) > 0 else np.array([], dtype=int)
    ppg_peaks_window = ppg_peaks_full[(ppg_peaks_full >= start_idx) & (ppg_peaks_full < end_idx)] - start_idx
    peaks_abs = peaks + start_idx

    # Find PTT based on the highly correlated segment if enabled
    shift_ppg_align = bool(inputs.get("shift_ppg_align", True))
    if shift_ppg_align:
        ptt_seconds, _seg = _calculate_high_corr_ptt(peaks_abs, ppg_peaks_window_ref, fs_proc, float(inputs["ref_fs"]), window_secs=min(15.0, window_size))
    else:
        ptt_seconds = 0.0
    
    # Store original unshifted peaks for physiological PTT computation
    original_ppg_window_ref = ppg_peaks_window_ref.copy()
    
    # Shift PPG peaks backward by PTT to align with SCG AO peaks
    ppg_peaks_window_ref = ppg_peaks_window_ref - int(ptt_seconds * float(inputs["ref_fs"]))
    ppg_peaks_window = ppg_peaks_window - int(ptt_seconds * fs_proc)
    
    # Filter out any peaks that become negative after shifting
    valid_ref_mask = ppg_peaks_window_ref >= 0
    ppg_peaks_window_ref = ppg_peaks_window_ref[valid_ref_mask]
    original_ppg_window_ref = original_ppg_window_ref[valid_ref_mask]
    
    valid_full_mask = ppg_peaks_window >= 0
    ppg_peaks_window = ppg_peaks_window[valid_full_mask]

    detection_metrics_window = None
    if len(peaks_abs) > 0 and len(ppg_peaks_window_ref) > 0:
        detection_metrics_window = compute_detection_metrics(peaks_abs, ppg_peaks_window_ref, fs_proc, float(inputs["ref_fs"]), tolerance_seconds=0.2, already_aligned=shift_ppg_align)

    comparison = None
    iqr_removed_ao_centers_s = np.array([])
    iqr_removed_ppg_centers_s = np.array([])
    if len(peaks) > 1 and len(ppg_peaks_window) > 1:
        (
            ao_intervals_ms,
            ppg_intervals_ms,
            ao_centers_s,
            ppg_centers_s,
            iqr_removed_ao_centers_s,
            iqr_removed_ppg_centers_s,
        ) = match_intervals_by_time(peaks, ppg_peaks_window, fs_proc, apply_iqr=bool(inputs["use_iqr_filter"]))
        if len(ao_intervals_ms) > 0:
            comparison = {
                "ao_intervals_ms": ao_intervals_ms,
                "ppg_intervals_ms": ppg_intervals_ms,
                "ao_centers_s": ao_centers_s,
                "ppg_centers_s": ppg_centers_s,
                "iqr_removed_ao_centers_s": iqr_removed_ao_centers_s,
                "iqr_removed_ppg_centers_s": iqr_removed_ppg_centers_s,
                "correlation": np.corrcoef(ao_intervals_ms, ppg_intervals_ms)[0, 1] if len(ao_intervals_ms) > 1 else np.nan,
                "rmse": np.sqrt(np.mean((ao_intervals_ms - ppg_intervals_ms) ** 2)),
                "mae": np.mean(np.abs(ao_intervals_ms - ppg_intervals_ms)),
                "mean_diff": np.mean(ao_intervals_ms - ppg_intervals_ms),
                "std_diff": np.std(ao_intervals_ms - ppg_intervals_ms),
                "mean_intervals": (ao_intervals_ms + ppg_intervals_ms) / 2,
                "diff_intervals": ao_intervals_ms - ppg_intervals_ms,
                "paper_metrics": compute_paper_metrics(ao_intervals_ms, ppg_intervals_ms),
                "ptt_metrics": compute_ptt_metrics(peaks_abs, original_ppg_window_ref, fs_proc, float(inputs["ref_fs"])),
            }

    return {
        "warning": None,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_time": start_time,
        "window_size": window_size,
        "time_axis": np.linspace(start_time, start_time + window_size, len(scg_window)) if len(scg_window) > 0 else np.array([]),
        "scg_window": scg_window,
        "scg_raw_window_resampled": scg_raw_window_resampled,
        "modes": modes,
        "omegas": omegas,
        "selected_idx": selected_idx,
        "center_freq_hz": center_freq_hz,
        "s_ao": s_ao,
        "s_ao_7": s_ao_7,
        "envelope": envelope,
        "smoothed_env": smoothed_env,
        "peaks": peaks,
        "peaks_abs": peaks_abs,
        "ppg_peaks_window": ppg_peaks_window,
        "ppg_peaks_window_ref": ppg_peaks_window_ref,
        "detection_metrics": detection_metrics_window,
        "comparison": comparison,
        "iqr_removed_ao_centers_s": iqr_removed_ao_centers_s,
        "iqr_removed_ppg_centers_s": iqr_removed_ppg_centers_s,
    }


def run_full_record_analysis(inputs: dict, progress_callback=None, status_callback=None) -> dict:
    scg_proc_full = inputs["scg_proc_full"]
    scg_raw = inputs["scg_raw"]
    fs_proc = float(inputs["fs_proc"])
    fs_infer = float(inputs["fs_infer"])
    trim_start = float(inputs["trim_start"])
    trim_end = float(inputs["trim_end"])
    beat_times_s = inputs["beat_times_s"]
    ppg_peaks_ref = inputs["ppg_peaks_ref"]
    ppg_peaks_full = inputs["ppg_peaks_full"]
    ppg_peak_times_s = inputs["ppg_peak_times_s"]
    use_iqr_filter = bool(inputs["use_iqr_filter"])
    show_sqa_overlay = bool(inputs["show_sqa_overlay"])
    exclude_bad_windows = bool(inputs["exclude_bad_windows"])
    bad_window_fraction_threshold = float(inputs["bad_window_fraction_threshold"])

    start_idx_global = int(trim_start * fs_proc)
    end_idx_global = int(trim_end * fs_proc)
    scg_proc_trim = scg_proc_full[start_idx_global:end_idx_global]
    scg_raw_trim = scg_raw[int(trim_start * fs_infer):int(trim_end * fs_infer)]
    ppg_peaks_full_trim = ppg_peaks_full[(ppg_peaks_full >= start_idx_global) & (ppg_peaks_full < end_idx_global)] - start_idx_global
    ppg_ref_mask = (beat_times_s >= trim_start) & (beat_times_s < trim_end)
    ppg_peaks_ref_trim = ppg_peaks_ref[ppg_ref_mask] if len(ppg_peaks_ref) > 0 else np.array([], dtype=int)

    total_duration = len(scg_proc_trim) / fs_proc if fs_proc > 0 else 0.0
    window_duration = 10.0
    num_windows = int(np.ceil(total_duration / window_duration)) if total_duration > 0 else 0

    sqa_result_full_record = None
    sample_bad_mask_full_record = None
    if show_sqa_overlay:
        sqa_result_full_record = combined_sqa_for_signal(
            scg_proc_trim,
            fs=fs_proc,
            segment_seconds=float(inputs["sqa_segment_seconds"]),
            min_flags=int(inputs["min_flags_to_reject"]),
            kurt_thresh=float(inputs["kurt_thresh"]),
            zcr_low=float(inputs["zcr_low"]),
            zcr_high=float(inputs["zcr_high"]),
            env_thresh=float(inputs["env_thresh"]),
            rms_low_percentile=int(inputs["rms_low_percentile"]),
            rms_high_percentile=int(inputs["rms_high_percentile"]),
            rms_low_mad_mult=float(inputs["rms_low_mad_mult"]),
            rms_high_mad_mult=float(inputs["rms_high_mad_mult"]),
        )
        sample_bad_mask_full_record = build_sample_bad_mask(len(scg_proc_trim), sqa_result_full_record)

    all_ao_peaks = []
    all_ao_intervals = []
    all_ao_intervals_times = []
    skipped_bad_windows = 0
    started = time.time()

    for i in range(num_windows):
        window_start = i * window_duration
        window_end = min(window_start + window_duration, total_duration)
        if status_callback:
            status_callback(f"Processing window {i + 1}/{num_windows} ({window_start:.1f}s - {window_end:.1f}s)")
        if progress_callback and num_windows > 0:
            progress_callback(int((i + 1) / num_windows * 100))

        start_idx_w = int(window_start * fs_proc)
        end_idx_w = int(window_end * fs_proc)
        if end_idx_w <= start_idx_w:
            continue

        if sample_bad_mask_full_record is not None and exclude_bad_windows:
            bad_frac_window = float(np.mean(sample_bad_mask_full_record[start_idx_w:end_idx_w]))
            if bad_frac_window >= bad_window_fraction_threshold:
                skipped_bad_windows += 1
                continue

        scg_window = scg_proc_trim[start_idx_w:end_idx_w]
        modes_w, omegas_w = svmd(scg_window, max_alpha=int(inputs["svmd_alpha"]), tau=0, stopc=3)
        if len(omegas_w) == 0:
            continue

        s_ao_w, _, _, _ = select_ao_modes(modes_w, omegas_w, fs_proc)
        _, _, _, ao_peaks_w = extract_ao_peaks(s_ao_w, fs_proc, float(inputs["prominence_factor"]), power=int(inputs["power_exp"]))
        if len(ao_peaks_w) > 0:
            ao_peaks_global = ao_peaks_w + start_idx_w
            all_ao_peaks.extend(ao_peaks_global.tolist())
            if len(ao_peaks_w) > 1:
                ao_intervals_w = np.diff(ao_peaks_w) / fs_proc * 1000.0
                ao_interval_times_w = (ao_peaks_w[:-1] + ao_peaks_w[1:]) / 2.0 / fs_proc + window_start + trim_start
                all_ao_intervals.extend(ao_intervals_w.tolist())
                all_ao_intervals_times.extend(ao_interval_times_w.tolist())

    elapsed_time = time.time() - started
    all_ao_peaks = np.asarray(all_ao_peaks, dtype=int)
    all_ao_intervals = np.asarray(all_ao_intervals, dtype=float)
    all_ao_intervals_times = np.asarray(all_ao_intervals_times, dtype=float)

    if sample_bad_mask_full_record is not None and show_sqa_overlay and len(all_ao_intervals_times) > 0:
        ao_interval_indices = (all_ao_intervals_times * fs_proc).astype(int)
        ao_good_mask = np.array([not sample_bad_mask_full_record[min(idx, len(sample_bad_mask_full_record) - 1)] for idx in ao_interval_indices], dtype=bool)
        all_ao_intervals = all_ao_intervals[ao_good_mask]
        all_ao_intervals_times = all_ao_intervals_times[ao_good_mask]

    save_messages = []
    file_label = inputs["file_label"]
    output_folder = inputs["output_folder"]
    if inputs["save_json_output"] and len(all_ao_peaks) > 0:
        global_peaks = all_ao_peaks + start_idx_global
        saved_file = save_peaks_to_json(global_peaks, fs_proc, file_label, output_folder)
        save_messages.append(f"AO Peaks saved to: {saved_file}")

    if inputs["save_ppg_json_output"] and inputs["beat_source"] == "Detect from PPG raw" and len(ppg_peak_times_s) > 0:
        ppg_peak_times = ppg_peak_times_s[(ppg_peak_times_s >= float(trim_start)) & (ppg_peak_times_s < float(trim_end))]
        if len(ppg_peak_times) > 0:
            saved_file_ppg = save_peaks_seconds_to_json(ppg_peak_times, file_label, output_folder, peak_label="PPG")
            save_messages.append(f"PPG Peaks saved to: {saved_file_ppg}")

    # Calculate PTT using the high-correlation segment scanner if enabled
    shift_ppg_align = bool(inputs.get("shift_ppg_align", True))
    ptt_best_segment = None
    if shift_ppg_align:
        ptt_seconds, ptt_best_segment = _calculate_high_corr_ptt(all_ao_peaks + start_idx_global, ppg_peaks_ref_trim, fs_proc, float(inputs["ref_fs"]))
    else:
        ptt_seconds = 0.0
    
    # Store original unshifted peaks for physiological PTT computation and rendering comparison
    original_ppg_peaks_ref_trim = ppg_peaks_ref_trim.copy()
    original_ppg_peaks_full_trim = ppg_peaks_full_trim.copy()
    
    # Shift PPG peaks backward by PTT to align with SCG AO peaks
    ppg_peaks_ref_trim = ppg_peaks_ref_trim - int(ptt_seconds * float(inputs["ref_fs"]))
    ppg_peaks_full_trim = ppg_peaks_full_trim - int(ptt_seconds * fs_proc)
    
    # Filter out any peaks that become negative or out of bounds after shifting
    valid_ref_mask = ppg_peaks_ref_trim >= 0
    ppg_peaks_ref_trim = ppg_peaks_ref_trim[valid_ref_mask]
    original_ppg_peaks_ref_trim = original_ppg_peaks_ref_trim[valid_ref_mask]
    
    valid_full_mask = (ppg_peaks_full_trim >= 0) & (ppg_peaks_full_trim < len(scg_proc_trim))
    ppg_peaks_full_trim = ppg_peaks_full_trim[valid_full_mask]
    original_ppg_peaks_full_trim = original_ppg_peaks_full_trim[valid_full_mask]

    ppg_intervals_full = np.diff(ppg_peaks_full_trim) / fs_proc * 1000.0 if len(ppg_peaks_full_trim) > 1 else np.array([])
    ppg_interval_times_full = (
        (ppg_peaks_full_trim[:-1] + ppg_peaks_full_trim[1:]) / 2.0 / fs_proc + trim_start if len(ppg_peaks_full_trim) > 1 else np.array([])
    )

    detection_metrics_full = None
    if len(all_ao_peaks) > 0 and len(ppg_peaks_ref_trim) > 0:
        detection_metrics_full = compute_detection_metrics(all_ao_peaks + start_idx_global, ppg_peaks_ref_trim, fs_proc, float(inputs["ref_fs"]), tolerance_seconds=0.2, already_aligned=shift_ppg_align)

    ao_intervals_matched = np.array([])
    ppg_intervals_matched = np.array([])
    ao_centers_s = np.array([])
    ppg_centers_s = np.array([])
    iqr_removed_ao_centers_s = np.array([])
    iqr_removed_ppg_centers_s = np.array([])
    if len(all_ao_peaks) > 1 and len(ppg_peaks_full_trim) > 1:
        (
            ao_intervals_matched,
            ppg_intervals_matched,
            ao_centers_s,
            ppg_centers_s,
            iqr_removed_ao_centers_s,
            iqr_removed_ppg_centers_s,
        ) = match_intervals_by_time(all_ao_peaks, ppg_peaks_full_trim, fs_proc, apply_iqr=use_iqr_filter)

    ptt_metrics = None
    paper_metrics = None
    correlation = np.nan
    rmse = np.nan
    mae = np.nan
    mean_diff = np.nan
    std_diff = np.nan
    mean_intervals = np.array([])
    diff_intervals = np.array([])
    if len(ao_intervals_matched) > 0:
        correlation = np.corrcoef(ao_intervals_matched, ppg_intervals_matched)[0, 1] if len(ao_intervals_matched) > 1 else np.nan
        rmse = np.sqrt(np.mean((ao_intervals_matched - ppg_intervals_matched) ** 2))
        mae = np.mean(np.abs(ao_intervals_matched - ppg_intervals_matched))
        mean_intervals = (ao_intervals_matched + ppg_intervals_matched) / 2
        diff_intervals = ao_intervals_matched - ppg_intervals_matched
        mean_diff = np.mean(diff_intervals)
        std_diff = np.std(diff_intervals)
        paper_metrics = compute_paper_metrics(ao_intervals_matched, ppg_intervals_matched)
        ptt_metrics = compute_ptt_metrics(all_ao_peaks + start_idx_global, original_ppg_peaks_ref_trim, fs_proc, float(inputs["ref_fs"]))

    full_time_axis = np.arange(len(scg_proc_trim)) / fs_proc + trim_start if fs_proc > 0 else np.array([])
    if len(scg_raw_trim) == len(scg_proc_trim):
        scg_raw_display = scg_raw_trim
    else:
        scg_raw_display = signal.resample(scg_raw_trim, len(scg_proc_trim)) if len(scg_raw_trim) > 0 else np.array([])

    return {
        "elapsed_time": elapsed_time,
        "skipped_bad_windows": skipped_bad_windows,
        "num_windows": num_windows,
        "all_ao_peaks": all_ao_peaks,
        "all_ao_intervals": all_ao_intervals,
        "all_ao_intervals_times": all_ao_intervals_times,
        "scg_proc_trim": scg_proc_trim,
        "scg_raw_display": scg_raw_display,
        "full_time_axis": full_time_axis,
        "ppg_intervals_full": ppg_intervals_full,
        "ppg_interval_times_full": ppg_interval_times_full,
        "ppg_peaks_full_trim": ppg_peaks_full_trim,
        "original_ppg_peaks_full_trim": original_ppg_peaks_full_trim,
        "ppg_peaks_ref_trim": ppg_peaks_ref_trim,
        "ptt_seconds": ptt_seconds,
        "ptt_best_segment": ptt_best_segment,
        "sample_bad_mask_full_record": sample_bad_mask_full_record,
        "sqa_result_full_record": sqa_result_full_record,
        "detection_metrics_full": detection_metrics_full,
        "ao_intervals_matched": ao_intervals_matched,
        "ppg_intervals_matched": ppg_intervals_matched,
        "ao_centers_s": ao_centers_s,
        "ppg_centers_s": ppg_centers_s,
        "iqr_removed_ao_centers_s": iqr_removed_ao_centers_s,
        "iqr_removed_ppg_centers_s": iqr_removed_ppg_centers_s,
        "paper_metrics": paper_metrics,
        "ptt_metrics": ptt_metrics,
        "correlation": correlation,
        "rmse": rmse,
        "mae": mae,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "mean_intervals": mean_intervals,
        "diff_intervals": diff_intervals,
        "save_messages": save_messages,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Subject metadata dialog
# ═══════════════════════════════════════════════════════════════════════════════

class SubjectDialog(QDialog):
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

        self._initials = QLineEdit(d.get('initials', ''))
        self._initials.setPlaceholderText("e.g.  AB")
        self._initials.setMaxLength(4)
        rx = QRegExp("[A-Za-z]{2,4}")
        self._initials.setValidator(QRegExpValidator(rx, self))
        self._initials.textChanged.connect(self._validate)
        form.addRow("Initials *", self._initials)

        self._age = QSpinBox()
        self._age.setRange(1, 120)
        self._age.setValue(int(d.get('age', 40)))
        form.addRow("Age (yrs) *", self._age)

        self._sex = QComboBox()
        self._sex.addItems(["Male", "Female", "Other / Not specified"])
        if d.get('sex') in ["Male", "Female", "Other / Not specified"]:
            self._sex.setCurrentText(d['sex'])
        form.addRow("Sex", self._sex)

        self._weight = QDoubleSpinBox()
        self._weight.setRange(1.0, 300.0)
        self._weight.setDecimals(1)
        self._weight.setSuffix(" kg")
        self._weight.setValue(float(d.get('weight_kg', 70.0)))
        self._weight.valueChanged.connect(self._update_bmi)
        form.addRow("Weight *", self._weight)

        self._height = QDoubleSpinBox()
        self._height.setRange(50.0, 250.0)
        self._height.setDecimals(1)
        self._height.setSuffix(" cm")
        self._height.setValue(float(d.get('height_cm', 170.0)))
        self._height.valueChanged.connect(self._update_bmi)
        form.addRow("Height *", self._height)

        self._bmi_lbl = QLabel()
        self._bmi_lbl.setObjectName("stat_value")
        self._bmi_lbl.setStyleSheet(f"color:{ACCENT};font-size:14px;font-weight:bold;")
        form.addRow("BMI", self._bmi_lbl)

        layout.addLayout(form)

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

        self._notes = QLineEdit(d.get('notes', ''))
        self._notes.setPlaceholderText("Optional free-text notes")
        layout.addWidget(QLabel("Notes"))
        layout.addWidget(self._notes)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._update_bmi()
        self._validate()
        self._on_normal_toggled(self._normal_cb.isChecked())

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


class ExportMlDialog(QDialog):
    def __init__(self, parent=None, patient_id: str = "", existing: dict | None = None):
        super().__init__(parent)
        self._patient_id = patient_id
        self.setWindowTitle("Export for ML")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._build_ui(existing or {})

    def _build_ui(self, d: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        hdr = QLabel("EXPORT SUBJECT FOR ML")
        hdr.setObjectName("section_title")
        layout.addWidget(hdr)

        pid = QLabel(f"Patient ID: {self._patient_id}")
        pid.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        layout.addWidget(pid)

        sep = QFrame(); sep.setObjectName("separator")
        layout.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        self._initials = QLineEdit(d.get("patient_initials", ""))
        self._initials.setPlaceholderText("Optional initials")
        self._initials.setMaxLength(6)
        form.addRow("Initials", self._initials)

        self._age = QSpinBox()
        self._age.setRange(0, 120)
        self._age.setValue(int(d.get("age", 0) or 0))
        form.addRow("Age (yrs)", self._age)

        self._sex = QComboBox()
        self._sex.addItems(["", "Male", "Female", "Other / Not specified"])
        existing_sex = d.get("sex", "")
        if existing_sex in ["Male", "Female", "Other / Not specified"]:
            self._sex.setCurrentText(existing_sex)
        form.addRow("Sex", self._sex)

        self._weight = QDoubleSpinBox()
        self._weight.setRange(0.0, 300.0)
        self._weight.setDecimals(1)
        self._weight.setSuffix(" kg")
        self._weight.setValue(float(d.get("weight_kg", 0.0) or 0.0))
        self._weight.valueChanged.connect(self._update_bmi)
        form.addRow("Weight", self._weight)

        self._height = QDoubleSpinBox()
        self._height.setRange(0.0, 250.0)
        self._height.setDecimals(1)
        self._height.setSuffix(" cm")
        self._height.setValue(float(d.get("height_cm", 0.0) or 0.0))
        self._height.valueChanged.connect(self._update_bmi)
        form.addRow("Height", self._height)

        self._bmi_lbl = QLabel()
        self._bmi_lbl.setObjectName("stat_value")
        self._bmi_lbl.setStyleSheet(f"color:{ACCENT};font-size:14px;font-weight:bold;")
        form.addRow("BMI", self._bmi_lbl)

        layout.addLayout(form)

        cond_group = QGroupBox("CARDIAC CONDITIONS")
        cond_layout = QVBoxLayout(cond_group)
        cond_layout.setSpacing(4)

        self._normal_cb = QCheckBox("Normal")
        self._normal_cb.toggled.connect(self._on_normal_toggled)
        cond_layout.addWidget(self._normal_cb)

        sep2 = QFrame(); sep2.setObjectName("separator")
        cond_layout.addWidget(sep2)

        self._cond_cbs: dict[str, QCheckBox] = {}
        for key in ["MS", "MR", "AR", "AS", "TR"]:
            cb = QCheckBox(key)
            cb.toggled.connect(self._on_condition_toggled)
            cond_layout.addWidget(cb)
            self._cond_cbs[key] = cb

        existing_conds = d.get("cardiac_conditions", [])
        if isinstance(existing_conds, str):
            existing_conds = [existing_conds]
        existing_conds = [str(c).strip().upper() for c in existing_conds if c]
        if "NORMAL" in existing_conds or not existing_conds:
            self._normal_cb.setChecked(True)
        else:
            for key, cb in self._cond_cbs.items():
                cb.setChecked(key in existing_conds)

        layout.addWidget(cond_group)

        self._notes = QLineEdit(d.get("notes", ""))
        self._notes.setPlaceholderText("Optional notes")
        layout.addWidget(QLabel("Notes"))
        layout.addWidget(self._notes)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._update_bmi()
        self._on_normal_toggled(self._normal_cb.isChecked())

    def _update_bmi(self):
        h_cm = self._height.value()
        h_m = h_cm / 100.0 if h_cm > 0 else 0.0
        bmi = self._weight.value() / (h_m * h_m) if h_m > 0 else 0.0
        self._bmi_lbl.setText(f"{bmi:.1f}" if bmi > 0 else "-")

    def _on_normal_toggled(self, checked: bool):
        for cb in self._cond_cbs.values():
            cb.setEnabled(not checked)
            if checked:
                cb.setChecked(False)

    def _on_condition_toggled(self, checked: bool):
        if checked:
            self._normal_cb.setChecked(False)

    def get_metadata(self) -> dict:
        h_cm = self._height.value()
        h_m = h_cm / 100.0 if h_cm > 0 else 0.0
        bmi = self._weight.value() / (h_m * h_m) if h_m > 0 else None
        conditions = [k for k, cb in self._cond_cbs.items() if cb.isChecked()]
        if self._normal_cb.isChecked() or not conditions:
            conditions = ["Normal"]

        initials = self._initials.text().strip().upper()
        sex = self._sex.currentText().strip()

        return {
            "patient_initials": initials,
            "age": self._age.value() if self._age.value() > 0 else None,
            "sex": sex if sex else None,
            "weight_kg": round(self._weight.value(), 1) if self._weight.value() > 0 else None,
            "height_cm": round(self._height.value(), 1) if self._height.value() > 0 else None,
            "bmi": round(bmi, 1) if bmi is not None else None,
            "cardiac_conditions": conditions,
            "notes": self._notes.text().strip(),
        }


class RawScgSvmdWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw SCG SVMD")
        self.resize(1600, 1000)

        self.df = None
        self.raw_df = None
        self.meta_data = None
        self.meta_path_display = ""
        self.current_csv_path = ""
        self.file_label = "uploaded"
        self.scg_df = pd.DataFrame()
        self.beats_df = pd.DataFrame()
        self.ppg_df = pd.DataFrame()
        self.scg_raw = np.array([], dtype=float)
        self.scg_proc_full = np.array([], dtype=float)
        self.fs_proc = 0.0
        self.fs_infer = 0.0
        self.actual_hz = 0.0
        self.duration_s = 0.0
        self.max_t = 0.0
        self.t0_ms = 0.0
        self.beat_times_s = np.array([], dtype=float)
        self.ppg_info = {}
        self.window_result = None
        self.full_result = None
        self.worker = None

        # ── Capture Buffers & State ───────────────────────────────────────────
        self._rb_scg_x       = RingBuffer(WINDOW_N)
        self._rb_scg_y       = RingBuffer(WINDOW_N)
        self._rb_scg_z       = RingBuffer(WINDOW_N)
        self._rb_scg_ts      = RingBuffer(WINDOW_N, dtype=np.int64)
        self._rb_scg_hts     = RingBuffer(WINDOW_N, dtype=np.int64)   # host clock
        self._rb_ppg         = RingBuffer(WINDOW_PPG_N)
        self._rb_ppg_ts      = RingBuffer(WINDOW_PPG_N, dtype=np.int64)
        self._rb_ppg_hts     = RingBuffer(WINDOW_PPG_N, dtype=np.int64)

        self._dq_scg_x   = deque(maxlen=WINDOW_N)
        self._dq_scg_y   = deque(maxlen=WINDOW_N)
        self._dq_scg_z   = deque(maxlen=WINDOW_N)
        self._dq_scg_ts  = deque(maxlen=WINDOW_N)
        self._dq_scg_hts = deque(maxlen=WINDOW_N)

        self._beat_ts:       list[int] = []
        self._beat_hts:      list[int] = []
        self._beat_intervals: deque    = deque(maxlen=BPM_HISTORY)
        self._last_beat_ts:  int | None = None
        self._last_beat_hts: int | None = None

        self._beat_line_cache:   list[list[tuple]] = [[], [], []]
        self._beat_ts_snapshot:  list[int]         = []

        self._scg_host_clock_ms: float | None = None
        self._ppg_host_clock_ms: float | None = None
        self._use_host_time      = False

        self._sample_count   = 0
        self._parse_err_count = 0
        self._rate_count     = 0

        self._plot_dirty    = False
        self._segment_dirty = False

        self._filter_on  = False
        self._notch_on   = False
        self._artifact_rejection_on = False
        
        self._bpf_sos    = butter(2, [BPF_LOW_HZ, BPF_HIGH_HZ],
                                  btype='bandpass', fs=SAMPLE_RATE, output='sos')
        from scipy.signal import iirnotch, tf2sos as _tf2sos
        _b, _a           = iirnotch(50.0, Q=30, fs=SAMPLE_RATE)
        self._notch_sos  = _tf2sos(_b, _a)
        self._reset_filter_state()
        
        self._ar_win    = SAMPLE_RATE
        self._ar_buf_x  = np.zeros(self._ar_win, dtype=np.float64)
        self._ar_buf_y  = np.zeros(self._ar_win, dtype=np.float64)
        self._ar_buf_z  = np.zeros(self._ar_win, dtype=np.float64)
        self._ar_idx    = 0
        self._ar_full   = False
        self._ar_prev   = None
        self._ar_rejected_count = 0

        self._thread: ReaderThread | None = None
        self._reader: SerialReader | None = None

        self._is_recording       = False
        self._async_writer: AsyncCSVWriter | None = None
        self._record_path        = ""
        self._record_samples     = 0
        self._record_first_ts: int | None = None
        self._record_last_ts:  int | None = None
        self._record_elapsed     = 0
        self._session_start_dt:  datetime | None = None

        self._subject: dict | None = None

        # ── Capture Timers ────────────────────────────────────────────────────
        self._ingest_timer = QTimer(self)
        self._ingest_timer.timeout.connect(self._drain_serial)

        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._on_plot_timer)
        self._plot_timer.start(40)   # 25 fps

        self._rate_timer = QTimer(self)
        self._rate_timer.timeout.connect(self._update_rate)

        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._tick_rec_timer)

        self._build_ui()
        self._connect_signals()
        self._refresh_workspace_files()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        sidebar_area = QScrollArea()
        sidebar_area.setWidgetResizable(True)
        sidebar_container = QWidget()
        self.sidebar_layout = QVBoxLayout(sidebar_container)
        self.sidebar_layout.setSpacing(12)
        self.sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_area.setWidget(sidebar_container)
        sidebar_area.setMinimumWidth(360)
        sidebar_area.setMaximumWidth(440)
        root_layout.addWidget(sidebar_area)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, stretch=1)

        self._build_sidebar_controls()
        self._build_tabs()

    def _build_sidebar_controls(self):
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_group)
        self.source_mode = QComboBox()
        self.source_mode.addItems(["Workspace CSV", "Open CSV"])
        source_layout.addWidget(self.source_mode)

        self.search_dir_edit = QLineEdit(DEFAULT_SEARCH_DIR)
        self.refresh_files_button = QPushButton("Refresh Workspace Files")
        source_layout.addWidget(QLabel("Search folder"))
        source_layout.addWidget(self.search_dir_edit)
        source_layout.addWidget(self.refresh_files_button)

        self.date_combo = QComboBox()
        source_layout.addWidget(QLabel("Workspace Date folder"))
        source_layout.addWidget(self.date_combo)

        self.file_combo = QComboBox()
        source_layout.addWidget(QLabel("Workspace CSV file"))
        source_layout.addWidget(self.file_combo)

        self.open_path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse CSV...")
        source_layout.addWidget(QLabel("Open CSV path"))
        source_layout.addWidget(self.open_path_edit)
        source_layout.addWidget(self.browse_button)

        self.load_button = QPushButton("Load Selected CSV")
        source_layout.addWidget(self.load_button)
        self.sidebar_layout.addWidget(source_group)

        params_group = QGroupBox("Analysis Settings")
        params_form = QFormLayout(params_group)
        self.expected_hz_spin = QDoubleSpinBox(); self.expected_hz_spin.setRange(1.0, 10000.0); self.expected_hz_spin.setValue(256.0); self.expected_hz_spin.setDecimals(2)
        self.override_fs_check = QCheckBox("Override inferred sample rate")
        self.override_hz_spin = QDoubleSpinBox(); self.override_hz_spin.setRange(1.0, 10000.0); self.override_hz_spin.setValue(256.0); self.override_hz_spin.setDecimals(2)
        self.preprocessing_mode = QComboBox(); self.preprocessing_mode.addItems(["MTI + Detrend + Median", "Bandpass 1-40 Hz", "None"]); self.preprocessing_mode.setCurrentText("Bandpass 1-40 Hz")
        self.target_fs_spin = QSpinBox(); self.target_fs_spin.setRange(50, 2000); self.target_fs_spin.setValue(256); self.target_fs_spin.setSingleStep(50)
        self.trim_start_spin = QDoubleSpinBox(); self.trim_start_spin.setRange(0.0, 1e9); self.trim_start_spin.setDecimals(3); self.trim_start_spin.setValue(0.0)
        self.trim_end_spin = QDoubleSpinBox(); self.trim_end_spin.setRange(0.0, 1e9); self.trim_end_spin.setDecimals(3); self.trim_end_spin.setValue(0.0)
        self.window_start_spin = QDoubleSpinBox(); self.window_start_spin.setRange(0.0, 1e9); self.window_start_spin.setDecimals(3); self.window_start_spin.setValue(0.0)
        self.window_size_spin = QDoubleSpinBox(); self.window_size_spin.setRange(0.5, 60.0); self.window_size_spin.setDecimals(2); self.window_size_spin.setValue(10.0)
        params_form.addRow("Expected sample rate (Hz)", self.expected_hz_spin)
        params_form.addRow(self.override_fs_check, self.override_hz_spin)
        params_form.addRow("Preprocessing", self.preprocessing_mode)
        params_form.addRow("Processing sample rate (Hz)", self.target_fs_spin)
        params_form.addRow("Trim start (s)", self.trim_start_spin)
        params_form.addRow("Trim end (s)", self.trim_end_spin)
        params_form.addRow("Window start (s)", self.window_start_spin)
        params_form.addRow("Window size (s)", self.window_size_spin)
        self.sidebar_layout.addWidget(params_group)

        sqa_group = QGroupBox("SQA / Thresholds")
        sqa_form = QFormLayout(sqa_group)
        self.show_sqa_overlay = QCheckBox("Enable SQA overlay"); self.show_sqa_overlay.setChecked(False)
        self.sqa_segment_spin = QSpinBox(); self.sqa_segment_spin.setRange(3, 5); self.sqa_segment_spin.setValue(4)
        self.min_flags_spin = QSpinBox(); self.min_flags_spin.setRange(1, 5); self.min_flags_spin.setValue(2)
        self.kurt_thresh_spin = QDoubleSpinBox(); self.kurt_thresh_spin.setRange(3.0, 20.0); self.kurt_thresh_spin.setValue(7.0); self.kurt_thresh_spin.setSingleStep(0.5)
        self.zcr_low_spin = QDoubleSpinBox(); self.zcr_low_spin.setRange(0.1, 2.0); self.zcr_low_spin.setValue(0.5); self.zcr_low_spin.setSingleStep(0.1)
        self.zcr_high_spin = QDoubleSpinBox(); self.zcr_high_spin.setRange(2.0, 20.0); self.zcr_high_spin.setValue(5.0); self.zcr_high_spin.setSingleStep(0.5)
        self.env_thresh_spin = QDoubleSpinBox(); self.env_thresh_spin.setRange(1.0, 5.0); self.env_thresh_spin.setValue(2.5); self.env_thresh_spin.setSingleStep(0.1)
        self.rms_low_percentile_spin = QSpinBox(); self.rms_low_percentile_spin.setRange(5, 30); self.rms_low_percentile_spin.setValue(20)
        self.rms_high_percentile_spin = QSpinBox(); self.rms_high_percentile_spin.setRange(70, 95); self.rms_high_percentile_spin.setValue(80)
        self.rms_low_mad_spin = QDoubleSpinBox(); self.rms_low_mad_spin.setRange(0.5, 4.0); self.rms_low_mad_spin.setValue(2.0); self.rms_low_mad_spin.setSingleStep(0.25)
        self.rms_high_mad_spin = QDoubleSpinBox(); self.rms_high_mad_spin.setRange(2.0, 8.0); self.rms_high_mad_spin.setValue(4.0); self.rms_high_mad_spin.setSingleStep(0.25)
        self.exclude_bad_windows_check = QCheckBox("Skip very noisy windows"); self.exclude_bad_windows_check.setChecked(True)
        self.use_iqr_filter_check = QCheckBox("Apply IQR outlier filtering"); self.use_iqr_filter_check.setChecked(True)
        self.bad_window_fraction_spin = QDoubleSpinBox(); self.bad_window_fraction_spin.setRange(0.30, 0.95); self.bad_window_fraction_spin.setValue(0.60); self.bad_window_fraction_spin.setSingleStep(0.05)
        for label, widget in [
            ("Enable SQA", self.show_sqa_overlay),
            ("SQA segment length (s)", self.sqa_segment_spin),
            ("Min flags to reject", self.min_flags_spin),
            ("Kurtosis threshold", self.kurt_thresh_spin),
            ("ZCR low bound (Hz)", self.zcr_low_spin),
            ("ZCR high bound (Hz)", self.zcr_high_spin),
            ("Envelope CV threshold", self.env_thresh_spin),
            ("RMS low percentile", self.rms_low_percentile_spin),
            ("RMS high percentile", self.rms_high_percentile_spin),
            ("RMS low MAD multiplier", self.rms_low_mad_spin),
            ("RMS high MAD multiplier", self.rms_high_mad_spin),
            ("Skip noisy windows", self.exclude_bad_windows_check),
            ("Apply IQR filter", self.use_iqr_filter_check),
            ("Skip if bad fraction >=", self.bad_window_fraction_spin),
        ]:
            sqa_form.addRow(label, widget)
        self.sidebar_layout.addWidget(sqa_group)

        svmd_group = QGroupBox("SVMD / PPG / Output")
        svmd_form = QFormLayout(svmd_group)
        self.svmd_alpha_spin = QSpinBox(); self.svmd_alpha_spin.setRange(100, 2000); self.svmd_alpha_spin.setValue(260)
        self.prominence_spin = QDoubleSpinBox(); self.prominence_spin.setRange(0.01, 0.30); self.prominence_spin.setValue(0.05); self.prominence_spin.setSingleStep(0.01)
        self.power_spin = QSpinBox(); self.power_spin.setRange(3, 9); self.power_spin.setValue(7)
        self.beat_source_combo = QComboBox(); self.beat_source_combo.addItems(["Use CSV beat_event", "Detect from PPG raw"]) ; self.beat_source_combo.setCurrentText("Detect from PPG raw")
        self.ppg_bp_low_spin = QDoubleSpinBox(); self.ppg_bp_low_spin.setRange(0.1, 10.0); self.ppg_bp_low_spin.setValue(0.5); self.ppg_bp_low_spin.setSingleStep(0.1)
        self.ppg_bp_high_spin = QDoubleSpinBox(); self.ppg_bp_high_spin.setRange(1.0, 20.0); self.ppg_bp_high_spin.setValue(8.0); self.ppg_bp_high_spin.setSingleStep(0.5)
        self.ppg_max_bpm_spin = QDoubleSpinBox(); self.ppg_max_bpm_spin.setRange(60.0, 300.0); self.ppg_max_bpm_spin.setValue(200.0); self.ppg_max_bpm_spin.setSingleStep(10.0)
        self.ppg_prom_spin = QDoubleSpinBox(); self.ppg_prom_spin.setRange(0.0, 1.0); self.ppg_prom_spin.setValue(0.3); self.ppg_prom_spin.setSingleStep(0.01)
        self.save_json_check = QCheckBox("Save AO Peaks to JSON")
        self.save_ppg_json_check = QCheckBox("Save PPG Peaks to JSON")
        self.shift_ppg_check = QCheckBox("Shift PPG peaks (PTT alignment)")
        self.shift_ppg_check.setChecked(True)
        self.output_folder_edit = QLineEdit("Saved_Peaks")
        for label, widget in [
            ("SVMD alpha", self.svmd_alpha_spin),
            ("Peak prominence", self.prominence_spin),
            ("Power exponent", self.power_spin),
            ("Beat reference", self.beat_source_combo),
            ("PPG bandpass low (Hz)", self.ppg_bp_low_spin),
            ("PPG bandpass high (Hz)", self.ppg_bp_high_spin),
            ("Max BPM for peak distance", self.ppg_max_bpm_spin),
            ("PPG peak prominence factor", self.ppg_prom_spin),
            ("Save AO JSON", self.save_json_check),
            ("Save PPG JSON", self.save_ppg_json_check),
            ("Shift PPG to SCG", self.shift_ppg_check),
            ("Output folder", self.output_folder_edit),
        ]:
            svmd_form.addRow(label, widget)
        self.sidebar_layout.addWidget(svmd_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.preview_button = QPushButton("Refresh Preview")
        self.run_window_button = QPushButton("Run Window Analysis")
        self.run_full_button = QPushButton("Analyze Full Record")
        actions_layout.addWidget(self.preview_button)
        actions_layout.addWidget(self.run_window_button)
        actions_layout.addWidget(self.run_full_button)
        self.batch_export_button = QPushButton("Batch Export >=")
        self.batch_min_seconds_spin = QDoubleSpinBox()
        self.batch_min_seconds_spin.setRange(1.0, 1e6)
        self.batch_min_seconds_spin.setDecimals(1)
        self.batch_min_seconds_spin.setValue(300.0)
        self.batch_min_seconds_spin.setSuffix(" s")
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_export_button)
        batch_row.addWidget(self.batch_min_seconds_spin)
        actions_layout.addLayout(batch_row)
        self.batch_full_button = QPushButton("Batch Full Analysis")
        self.batch_save_csv_check = QCheckBox("Save recap CSV")
        self.batch_save_csv_check.setChecked(False)
        self.batch_full_status = QLabel("")
        self.batch_full_status.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        actions_layout.addWidget(self.batch_full_button)
        actions_layout.addWidget(self.batch_save_csv_check)
        actions_layout.addWidget(self.batch_full_status)
        self.batch_export_status = QLabel("")
        self.batch_export_status.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        actions_layout.addWidget(self.batch_export_status)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Ready")
        actions_layout.addWidget(self.progress_bar)
        actions_layout.addWidget(self.status_label)
        self.sidebar_layout.addWidget(actions_group)
        self.sidebar_layout.addStretch(1)

    def _build_tabs(self):
        self.capture_tab = QWidget()
        self.overview_tab = QWidget()
        self.window_tab = QWidget()
        self.full_tab = QWidget()
        self.batch_tab = QWidget()
        self.logs_tab = QWidget()
        self.tabs.addTab(self.capture_tab, "Real-time Capture")
        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.addTab(self.window_tab, "Window Analysis")
        self.tabs.addTab(self.full_tab, "Full Record")
        self.tabs.addTab(self.batch_tab, "Batch Recap")
        self.tabs.addTab(self.logs_tab, "Logs")

        # ── Capture Tab Layout ────────────────────────────────────────────────
        capture_layout = QHBoxLayout(self.capture_tab)
        capture_layout.setContentsMargins(10, 10, 10, 10)
        capture_layout.setSpacing(10)
        capture_layout.addWidget(self._build_plots_panel(), stretch=1)
        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(230)
        capture_layout.addWidget(sidebar)

        overview_hbox = QHBoxLayout(self.overview_tab)
        overview_hbox.setContentsMargins(10, 10, 10, 10)
        overview_hbox.setSpacing(12)

        self.workspace_summary_view = QTextBrowser()
        self.workspace_summary_view.setReadOnly(True)
        self.workspace_summary_view.setOpenLinks(False)
        self.workspace_summary_view.setMinimumWidth(680)
        overview_hbox.addWidget(self.workspace_summary_view, stretch=3)

        active_panel = QWidget()
        active_layout = QVBoxLayout(active_panel)
        active_layout.setContentsMargins(0, 0, 0, 0)
        active_layout.setSpacing(8)

        metrics_row = QHBoxLayout()
        self.metric_samples = QLabel("SCG Samples: -")
        self.metric_beats = QLabel("PPG Beats: -")
        self.metric_rate = QLabel("Actual Rate: -")
        self.metric_error = QLabel("Rate Error: -")
        for widget in [self.metric_samples, self.metric_beats, self.metric_rate, self.metric_error]:
            metrics_row.addWidget(widget)
        metrics_row.addStretch(1)
        active_layout.addLayout(metrics_row)

        self.metadata_text = QTextBrowser(); self.metadata_text.setReadOnly(True)
        self.metadata_text.setMinimumHeight(200)
        self.metadata_text.setMaximumHeight(280)
        active_layout.addWidget(self.metadata_text)

        self.ppg_preview_plot = MplPlotWidget("Processed PPG and Detected Peaks", width=9, height=3.5, dpi=100)
        active_layout.addWidget(self.ppg_preview_plot, stretch=1)

        overview_hbox.addWidget(active_panel, stretch=2)

        batch_layout = QVBoxLayout(self.batch_tab)
        batch_layout.setContentsMargins(10, 10, 10, 10)
        batch_layout.setSpacing(8)
        batch_layout.addWidget(QLabel("Batch Full Analysis Recap"))
        self.batch_recap_view = QTextBrowser()
        self.batch_recap_view.setReadOnly(True)
        self.batch_recap_view.setOpenLinks(False)
        batch_layout.addWidget(self.batch_recap_view)

        window_root_layout = QVBoxLayout(self.window_tab)
        self.window_scroll_area = QScrollArea()
        self.window_scroll_area.setWidgetResizable(True)
        self.window_scroll_area.setFrameShape(QFrame.NoFrame)
        self.window_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        window_root_layout.addWidget(self.window_scroll_area)

        window_container = QWidget()
        window_layout = QVBoxLayout(window_container)
        window_layout.setSpacing(18)
        window_layout.setContentsMargins(4, 4, 12, 12)
        self.window_scroll_area.setWidget(window_container)

        self.window_raw_plot = MplPlotWidget("1. Raw SCG Signal", width=11, height=3.0, dpi=100)
        self.window_filtered_plot = MplPlotWidget("2. Filtered SCG Signal", width=11, height=3.0, dpi=100, sharex=self.window_raw_plot.axes)
        self.window_decomposed_plot = MplPlotWidget("3. SVMD Decomposed Modes", width=11, height=5.0, dpi=100, sharex=self.window_raw_plot.axes)
        self.window_reconstructed_plot = MplPlotWidget("4. Reconstructed AO Signal", width=11, height=3.0, dpi=100, sharex=self.window_raw_plot.axes)
        self.window_power7_plot = MplPlotWidget("5. Reconstructed Signal after 7th Power", width=11, height=3.0, dpi=100, sharex=self.window_raw_plot.axes)
        self.window_detected_power7_plot = MplPlotWidget("6. Detected Peaks at 7th Power (Envelope)", width=11, height=3.0, dpi=100, sharex=self.window_raw_plot.axes)
        self.window_peaks_original_plot = MplPlotWidget("7. Original SCG Signal with Detected Peaks", width=11, height=3.0, dpi=100, sharex=self.window_raw_plot.axes)

        self.window_metrics_table = QTableWidget()
        self.window_metrics_table.setMinimumHeight(180)

        window_layout.addWidget(self.window_raw_plot)
        window_layout.addWidget(self.window_filtered_plot)
        window_layout.addWidget(self.window_decomposed_plot)
        window_layout.addWidget(self.window_reconstructed_plot)
        window_layout.addWidget(self.window_power7_plot)
        window_layout.addWidget(self.window_detected_power7_plot)
        window_layout.addWidget(self.window_peaks_original_plot)
        window_layout.addWidget(self.window_metrics_table)

        full_root_layout = QVBoxLayout(self.full_tab)
        self.full_scroll_area = QScrollArea()
        self.full_scroll_area.setWidgetResizable(True)
        self.full_scroll_area.setFrameShape(QFrame.NoFrame)
        self.full_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        full_root_layout.addWidget(self.full_scroll_area)

        full_container = QWidget()
        self.full_layout = QVBoxLayout(full_container)
        self.full_layout.setSpacing(18)
        self.full_layout.setContentsMargins(4, 4, 12, 12)
        self.full_scroll_area.setWidget(full_container)

        self.full_raw_plot = MplPlotWidget("Raw SCG with AO Peaks", width=11, height=4.0, dpi=100)
        self.full_proc_plot = MplPlotWidget("Processed SCG with AO Peaks and PPG Beats", width=11, height=4.0, dpi=100, sharex=self.full_raw_plot.axes)
        self.full_ppg_plot = MplPlotWidget("Processed PPG with Peaks", width=11, height=4.0, dpi=100, sharex=self.full_raw_plot.axes)
        self.full_interval_plot = MplPlotWidget("PPG-PPG and AO-AO Intervals", width=11, height=4.0, dpi=100, sharex=self.full_raw_plot.axes)
        self.full_ba_plot = MplPlotWidget("Bland-Altman Plot", width=11, height=4.0, dpi=100)
        self.full_corr_plot = MplPlotWidget("AO-AO vs PPG-PPG Intervals", width=11, height=4.0, dpi=100)
        self.full_shift_plot = MplPlotWidget("PPG Peak Shift Alignment (Before vs After)", width=11, height=4.0, dpi=100)
        self.full_metrics_table = QTableWidget(); self.full_metrics_table.setMinimumHeight(170)
        self.detect_metrics_table = QTableWidget(); self.detect_metrics_table.setMinimumHeight(170)
        self.detection_errors_table = QTableWidget(); self.detection_errors_table.setMinimumHeight(170)
        self.iqr_table = QTableWidget(); self.iqr_table.setMinimumHeight(150)
        for plot in [self.full_raw_plot, self.full_proc_plot, self.full_ppg_plot, self.full_interval_plot, self.full_ba_plot, self.full_corr_plot, self.full_shift_plot]:
            self.full_layout.addWidget(plot)
        self.full_layout.addWidget(QLabel("Interval Metrics"))
        self.full_layout.addWidget(self.full_metrics_table)
        self.full_layout.addWidget(QLabel("Detection Metrics"))
        self.full_layout.addWidget(self.detect_metrics_table)
        self.full_layout.addWidget(QLabel("Detection Errors"))
        self.full_layout.addWidget(self.detection_errors_table)
        self.full_layout.addWidget(QLabel("Removed Intervals"))
        self.full_layout.addWidget(self.iqr_table)

        export_group = QGroupBox("ML Export")
        export_layout = QHBoxLayout(export_group)
        export_layout.setSpacing(10)
        self.export_ml_button = QPushButton("Export for ML")
        self.export_ml_button.setEnabled(False)
        self.export_ml_status = QLabel("")
        self.export_ml_status.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        export_layout.addWidget(self.export_ml_button)
        export_layout.addWidget(self.export_ml_status, stretch=1)
        self.full_layout.addWidget(export_group)

        self.full_layout.addStretch(1)

        logs_layout = QVBoxLayout(self.logs_tab)
        self.log_text = QPlainTextEdit(); self.log_text.setReadOnly(True)
        logs_layout.addWidget(self.log_text)

    def _build_plots_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

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

        seg_hdr = QLabel("LAST BEAT SEGMENT")
        seg_hdr.setObjectName("section_title")
        seg_hdr.setStyleSheet(f"color:{AMBER};font-size:11px;font-weight:bold;letter-spacing:3px;")
        layout.addWidget(seg_hdr)

        self._segment_plot = make_plot_widget("BEAT SEGMENT")
        self._segment_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._segment_plot.getAxis('bottom').setLabel('time (s)')
        self._segment_plot.showGrid(x=True, y=True, alpha=0.15)
        self._segment_curves: list[pg.PlotDataItem] = []
        for color in COLORS_SCG:
            self._segment_curves.append(
                self._segment_plot.plot([], [],
                    pen=pg.mkPen(color, width=1.5, cosmetic=True))
            )
        layout.addWidget(self._segment_plot)

        return w

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

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

        stats_card = QFrame(); stats_card.setObjectName("card")
        sl = QVBoxLayout(stats_card); sl.setContentsMargins(14, 12, 14, 12); sl.setSpacing(8)
        self._stat_beats = self._stat_row(sl, "BEATS",        "0")
        self._stat_rate  = self._stat_row(sl, "SAMPLE RATE",  "-- Hz")
        self._stat_lost  = self._stat_row(sl, "PARSE ERRORS", "0")
        self._stat_ar    = self._stat_row(sl, "REJECTED",     "0")
        layout.addWidget(stats_card)

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

        cfg_card = QFrame(); cfg_card.setObjectName("card")
        cfg_l = QVBoxLayout(cfg_card); cfg_l.setContentsMargins(14, 12, 14, 12); cfg_l.setSpacing(6)
        cfg_hdr = QLabel("SETTINGS"); cfg_hdr.setObjectName("stat_label")
        cfg_l.addWidget(cfg_hdr)
        self._filter_cb = QCheckBox("BANDPASS 0.5–50 Hz")
        self._filter_cb.toggled.connect(self._on_filter_toggled)
        cfg_l.addWidget(self._filter_cb)
        self._notch_cb = QCheckBox("NOTCH 50 Hz (mains)")
        self._notch_cb.setEnabled(False)
        self._notch_cb.toggled.connect(self._on_notch_toggled)
        cfg_l.addWidget(self._notch_cb)
        self._artifact_cb = QCheckBox("ARTIFACT REJECTION")
        self._artifact_cb.toggled.connect(self._on_artifact_toggled)
        cfg_l.addWidget(self._artifact_cb)
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

    def _connect_signals(self):
        self.refresh_files_button.clicked.connect(self._refresh_workspace_files)
        self.browse_button.clicked.connect(self._browse_csv)
        self.load_button.clicked.connect(self._load_selected_csv)
        self.preview_button.clicked.connect(self._refresh_preview)
        self.run_window_button.clicked.connect(self._run_window_analysis)
        self.run_full_button.clicked.connect(self._run_full_analysis)
        self.export_ml_button.clicked.connect(self._export_full_record_for_ml)
        self.batch_export_button.clicked.connect(self._batch_export_for_ml)
        self.batch_full_button.clicked.connect(self._batch_full_record_analysis)
        self.source_mode.currentIndexChanged.connect(self._update_source_controls)
        self.beat_source_combo.currentIndexChanged.connect(self._update_ppg_json_toggle)
        self.date_combo.currentIndexChanged.connect(self._on_date_changed)
        self.workspace_summary_view.anchorClicked.connect(self._on_summary_link_clicked)
        self._update_source_controls()
        self._update_ppg_json_toggle()

    def _log(self, message: str):
        self.log_text.appendPlainText(message)

    def _set_status(self, message: str):
        self.status_label.setText(message)
        self._log(message)

    def _update_source_controls(self):
        workspace_mode = self.source_mode.currentText() == "Workspace CSV"
        self.search_dir_edit.setEnabled(workspace_mode)
        self.refresh_files_button.setEnabled(workspace_mode)
        self.date_combo.setEnabled(workspace_mode)
        self.file_combo.setEnabled(workspace_mode)
        self.open_path_edit.setEnabled(not workspace_mode)
        self.browse_button.setEnabled(not workspace_mode)

    def _update_ppg_json_toggle(self):
        self.save_ppg_json_check.setEnabled(self.beat_source_combo.currentText() == "Detect from PPG raw")

    def _refresh_workspace_files(self):
        search_dir = Path(self.search_dir_edit.text().strip() or DEFAULT_SEARCH_DIR)
        self.date_combo.clear()
        self.file_combo.clear()
        self.date_to_files = {}
        if not search_dir.exists() or not search_dir.is_dir():
            self._set_status("Workspace folder does not exist.")
            return
        candidates = sorted(search_dir.rglob("*.csv"))
        if not candidates:
            self._set_status("No CSV files found in workspace folder.")
            return

        for path in candidates:
            date_folder = path.parent.name
            if not date_folder or date_folder == search_dir.name:
                date_folder = "Root"
            if date_folder not in self.date_to_files:
                self.date_to_files[date_folder] = []
            self.date_to_files[date_folder].append(path)

        self.date_combo.blockSignals(True)
        for d in sorted(self.date_to_files.keys()):
            self.date_combo.addItem(d, d)
        self.date_combo.blockSignals(False)

        if self.date_combo.count() > 0:
            self.date_combo.setCurrentIndex(0)
            self._on_date_changed()

        self._set_status(f"Found {len(candidates)} CSV files across {len(self.date_to_files)} date folders.")
        self._update_workspace_summary(candidates)

    def _update_workspace_summary(self, csv_paths: list[Path]):
        subjects = {}
        total_time = 0.0
        total_recordings = len(csv_paths)

        all_ages = []
        male_count = 0
        female_count = 0
        conditions_count = {}

        for path in csv_paths:
            duration = _get_csv_duration(path)
            total_time += duration

            meta, _ = _read_metadata_for_csv(str(path))

            initials = "N/A"
            sex = "N/A"
            age = "N/A"
            conds = ["Normal"]

            if meta:
                initials = meta.get("patient_initials", "N/A").strip()
                sex = meta.get("sex", "N/A").strip()
                age_val = meta.get("age", None)
                if isinstance(age_val, (int, float)):
                    age = int(age_val)
                conds = meta.get("cardiac_conditions", ["Normal"])
                if not isinstance(conds, list):
                    conds = [str(conds)]

            if initials == "N/A" or not initials:
                stem_parts = path.stem.split("_")
                if stem_parts:
                    initials = stem_parts[0]

            initials = initials.upper()

            if initials not in subjects:
                subjects[initials] = {
                    "initials": initials,
                    "sex": sex,
                    "age": age,
                    "conditions": set(),
                    "recordings": [],
                    "total_duration": 0.0
                }

                if sex.lower() == "male":
                    male_count += 1
                elif sex.lower() == "female":
                    female_count += 1

                if isinstance(age, int):
                    all_ages.append(age)

            subjects[initials]["conditions"].update(conds)
            subjects[initials]["total_duration"] += duration
            subjects[initials]["recordings"].append({
                "path": str(path),
                "name": path.name,
                "date": path.parent.name,
                "duration": duration
            })

            for c in conds:
                conditions_count[c] = conditions_count.get(c, 0) + 1

        total_subjects = len(subjects)
        avg_age = sum(all_ages) / len(all_ages) if all_ages else 0.0
        min_age = min(all_ages) if all_ages else 0
        max_age = max(all_ages) if all_ages else 0

        total_hours = int(total_time // 3600)
        total_mins = int((total_time % 3600) // 60)
        total_secs = total_time % 60
        if total_hours > 0:
            total_time_str = f"{total_hours}h {total_mins}m {total_secs:.1f}s"
        elif total_mins > 0:
            total_time_str = f"{total_mins}m {total_secs:.1f}s"
        else:
            total_time_str = f"{total_secs:.1f} seconds"

        conds_list = []
        for cond, count in sorted(conditions_count.items(), key=lambda x: x[1], reverse=True):
            conds_list.append(f"{cond}: {count}")
        conditions_breakdown = ", ".join(conds_list) if conds_list else "None"

        subjects_rows = ""
        for initials, sub in sorted(subjects.items()):
            cond_badges = ""
            for c in sorted(sub["conditions"]):
                style = "background-color: #c6f6d5; color: #22543d;" if c.lower() == "normal" else "background-color: #fed7d7; color: #9b2c2c;"
                cond_badges += f'<span style="border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: bold; margin-right: 4px; {style}">{c}</span>'

            demo_str = f"{sub['sex']}, {sub['age']} yrs" if sub['sex'] != "N/A" and sub['age'] != "N/A" else "Unknown Demographics"

            rec_links = ""
            for r in sorted(sub["recordings"], key=lambda x: (x["date"], x["name"])):
                rec_links += f"""
                <a href="load:{r['path']}" style="text-decoration: none; display: inline-block; background-color: #ffffff; border: 1px solid #cbd5e0; border-radius: 4px; padding: 3px 8px; font-size: 11px; color: #2b6cb0; font-weight: 600; margin-right: 6px; margin-bottom: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    📁 {r['date']}/{r['name']} ({r['duration']:.1f}s)
                </a>
                """

            subjects_rows += f"""
            <tr style="border-bottom: 1px solid #edf2f7;">
                <td style="padding: 10px; font-weight: bold; color: #1a365d; font-size: 13px;">👤 {initials}</td>
                <td style="padding: 10px; color: #4a5568;">{demo_str}</td>
                <td style="padding: 10px;">{cond_badges}</td>
                <td style="padding: 10px; text-align: center; font-weight: bold; color: #2d3748;">{len(sub['recordings'])}</td>
                <td style="padding: 10px; text-align: right; font-weight: bold; color: #dd6b20;">{sub['total_duration']:.1f} s</td>
            </tr>
            <tr style="background-color: #f7fafc; border-bottom: 1px solid #edf2f7;">
                <td colspan="5" style="padding: 8px 16px;">
                    <div style="font-size: 10px; color: #718096; margin-bottom: 5px; font-weight: 600;">Load Recording:</div>
                    <div style="line-height: 1.6;">
                        {rec_links}
                    </div>
                </td>
            </tr>
            """

        html = f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8f9fa; color: #2d3748; padding: 10px; margin: 0;">
            <div style="font-size: 17px; font-weight: bold; color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 6px; margin-bottom: 12px;">📊 Workspace Overview Dashboard</div>

            <!-- Stats Breakdown Cards Grid -->
            <table style="width: 100%; border-collapse: separate; border-spacing: 10px; margin-bottom: 15px;">
                <tr>
                    <!-- Demographic Summary Card -->
                    <td style="width: 50%; vertical-align: top; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <div style="font-weight: bold; color: #2b6cb0; margin-bottom: 8px; font-size: 13px;">👥 Demographic Summary</div>
                        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                            <tr>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; color: #4a5568;">Total Unique Subjects</td>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; text-align: right; font-weight: bold; color: #1a202c;">{total_subjects}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; color: #4a5568;">Gender Breakdown</td>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; text-align: right; font-weight: 500;">Male: <span style="color: #2b6cb0; font-weight: bold;">{male_count}</span> | Female: <span style="color: #b83280; font-weight: bold;">{female_count}</span></td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 0; color: #4a5568;">Age Distribution</td>
                                <td style="padding: 5px 0; text-align: right; font-weight: 500;">Avg: <span style="color: #2d3748; font-weight: bold;">{avg_age:.1f}</span> yrs (Range: {min_age} - {max_age})</td>
                            </tr>
                        </table>
                    </td>

                    <!-- Conditions & Records Summary Card -->
                    <td style="width: 50%; vertical-align: top; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <div style="font-weight: bold; color: #2b6cb0; margin-bottom: 8px; font-size: 13px;">📁 Clinical & Recording Stats</div>
                        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                            <tr>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; color: #4a5568;">Total Recordings</td>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; text-align: right; font-weight: bold; color: #1a202c;">{total_recordings}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; color: #4a5568;">Total Cumulative Time</td>
                                <td style="padding: 5px 0; border-bottom: 1px solid #edf2f7; text-align: right; font-weight: bold; color: #dd6b20;">{total_time_str}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px 0; color: #4a5568;">Cardiac Conditions</td>
                                <td style="padding: 5px 0; text-align: right; font-size: 11px; font-weight: 600; color: #2d3748;">{conditions_breakdown}</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>

            <!-- Subjects List Table -->
            <div style="font-weight: bold; color: #1a365d; margin-bottom: 8px; font-size: 14px; border-bottom: 1px solid #cbd5e0; padding-bottom: 4px;">📂 Subject Recordings Directory</div>
            <table style="width: 100%; border-collapse: collapse; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;">
                <thead>
                    <tr style="background-color: #ebf8ff; color: #2b6cb0; text-align: left; border-bottom: 2px solid #bee3f8; font-size: 12px;">
                        <th style="padding: 10px;">Subject</th>
                        <th style="padding: 10px;">Demographics</th>
                        <th style="padding: 10px;">Conditions</th>
                        <th style="padding: 10px; text-align: center;">Recordings</th>
                        <th style="padding: 10px; text-align: right;">Total Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {subjects_rows}
                </tbody>
            </table>
        </body>
        </html>
        """
        self.workspace_summary_view.setHtml(html)

    def _on_date_changed(self):
        self.file_combo.clear()
        selected_date = self.date_combo.currentText()
        if not selected_date or not hasattr(self, "date_to_files") or selected_date not in self.date_to_files:
            return
        files = self.date_to_files[selected_date]
        for path in sorted(files, key=lambda p: p.name):
            self.file_combo.addItem(path.name, str(path))

    def _on_summary_link_clicked(self, url):
        path_str = url.toString()
        if path_str.startswith("load:"):
            csv_path = path_str[5:]
            self._load_recording_by_path(csv_path)

    def _load_recording_by_path(self, csv_path: str):
        path_obj = Path(csv_path)
        if not path_obj.exists():
            QMessageBox.warning(self, "File Not Found", f"The CSV file does not exist:\n{csv_path}")
            return

        self.source_mode.setCurrentText("Workspace CSV")
        self._update_source_controls()

        date_folder = path_obj.parent.name
        search_dir = Path(self.search_dir_edit.text().strip() or DEFAULT_SEARCH_DIR)
        if not date_folder or date_folder == search_dir.name:
            date_folder = "Root"

        index_date = self.date_combo.findText(date_folder)
        if index_date >= 0:
            self.date_combo.blockSignals(True)
            self.date_combo.setCurrentIndex(index_date)
            self.date_combo.blockSignals(False)
            self._on_date_changed()

        index_file = self.file_combo.findData(str(path_obj))
        if index_file < 0:
            index_file = self.file_combo.findText(path_obj.name)

        if index_file >= 0:
            self.file_combo.setCurrentIndex(index_file)

        self._load_selected_csv()

    def _browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", str(Path.cwd()), "CSV Files (*.csv)")
        if file_path:
            self.open_path_edit.setText(file_path)

    def _selected_csv_path(self):
        if self.source_mode.currentText() == "Workspace CSV":
            if self.file_combo.count() == 0:
                return ""
            return self.file_combo.currentData() or self.file_combo.currentText()
        return self.open_path_edit.text().strip()

    def _load_selected_csv(self):
        csv_path = self._selected_csv_path()
        if not csv_path:
            QMessageBox.information(self, "Select CSV", "Choose a CSV file first.")
            return
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "Missing file", f"CSV file not found:\n{csv_path}")
            return

        try:
            raw_df = _read_csv_from_source(None, csv_path)
            df = _prepare_df(raw_df)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Failed to load CSV:\n{exc}")
            return

        self.raw_df = raw_df
        self.df = df
        self.current_csv_path = csv_path
        self.file_label = Path(csv_path).stem
        self.full_result = None
        if hasattr(self, "export_ml_button"):
            self.export_ml_button.setEnabled(False)
            self.export_ml_status.setText("")
        if hasattr(self, "batch_export_status"):
            self.batch_export_status.setText("")

        try:
            self.meta_data, self.meta_path_display = _read_metadata_for_csv(csv_path)
        except Exception as exc:
            self.meta_data = None
            self.meta_path_display = ""
            self._log(f"Metadata read failed: {exc}")

        self.scg_df = df[df[["x_g", "y_g", "z_g"]].notna().any(axis=1)].copy()
        self.beats_df = df[df["beat_event"] == 1].copy()
        self.ppg_df = df[df[OPTIONAL_PPG_COL].notna()].copy() if OPTIONAL_PPG_COL in df.columns else pd.DataFrame()
        if self.scg_df.empty:
            QMessageBox.warning(self, "No SCG data", "No SCG rows found (x_g/y_g/z_g are empty for all rows).")
            return

        self.duration_s, self.actual_hz = _compute_rate(self.scg_df)
        self.fs_infer = self.override_hz_spin.value() if self.override_fs_check.isChecked() else self.actual_hz
        if self.fs_infer <= 0:
            QMessageBox.warning(self, "Invalid rate", "Invalid sampling rate inferred. Check timestamp_ms.")
            return

        self.t0_ms = float(self.scg_df["timestamp_ms"].iloc[0])
        self.scg_df["time_s"] = (self.scg_df["timestamp_ms"] - self.t0_ms) / 1000.0
        self.beats_df["time_s"] = (self.beats_df["timestamp_ms"] - self.t0_ms) / 1000.0 if not self.beats_df.empty else np.array([])
        if not self.ppg_df.empty:
            self.ppg_df["time_s"] = (self.ppg_df["timestamp_ms"] - self.t0_ms) / 1000.0

        self.max_t = float(self.scg_df["time_s"].iloc[-1])
        self.trim_end_spin.setMaximum(max(self.max_t, 0.0))
        self.trim_end_spin.setValue(self.max_t)
        self.window_start_spin.setMaximum(max(self.max_t, 0.0))
        self.window_size_spin.setMaximum(max(0.5, self.max_t))
        self.window_size_spin.setValue(min(10.0, max(0.5, self.max_t)))
        self.trim_start_spin.setMaximum(max(self.max_t, 0.0))

        self.metric_samples.setText(f"SCG Samples: {len(self.scg_df):,}")
        self.metric_beats.setText(f"PPG Beats: {len(self.beats_df):,}")
        self.metric_rate.setText(f"Actual Rate: {self.actual_hz:.2f} Hz")
        diff_hz = self.actual_hz - self.expected_hz_spin.value()
        pct = (diff_hz / self.expected_hz_spin.value() * 100.0) if self.expected_hz_spin.value() > 0 else 0.0
        self.metric_error.setText(f"Rate Error: {diff_hz:+.2f} Hz ({pct:+.2f}%)")

        self.metadata_text.setHtml(_generate_metadata_html(self.meta_data, self.duration_s))

        self._refresh_preview()
        self._set_status(f"Loaded {Path(csv_path).name}")

    def _base_params(self):
        return {
            "beat_source": self.beat_source_combo.currentText(),
            "ppg_bp_low": self.ppg_bp_low_spin.value(),
            "ppg_bp_high": self.ppg_bp_high_spin.value(),
            "ppg_max_bpm": self.ppg_max_bpm_spin.value(),
            "ppg_prom": self.ppg_prom_spin.value(),
            "fs_proc": float(self.target_fs_spin.value()),
            "shift_ppg_align": self.shift_ppg_check.isChecked(),
        }

    def _build_processing_state(self):
        if self.scg_df.empty:
            return None
        self.scg_raw = self.scg_df["z_g"].to_numpy(dtype=float)
        self.scg_proc_full, self.fs_proc = _processing_signal(
            self.scg_raw,
            self.fs_infer,
            float(self.target_fs_spin.value()),
            self.preprocessing_mode.currentText(),
        )
        self.beat_times_s = self.beats_df["time_s"].to_numpy(dtype=float) if len(self.beats_df) > 0 else np.array([], dtype=float)
        self.ppg_info = _current_ppg_info(
            self.ppg_df,
            self.beat_times_s,
            self.beat_source_combo.currentText(),
            self.ppg_bp_low_spin.value(),
            self.ppg_bp_high_spin.value(),
            self.ppg_max_bpm_spin.value(),
            self.ppg_prom_spin.value(),
            self.fs_proc,
        )
        return self.ppg_info

    def _refresh_preview(self):
        if self.scg_df.empty:
            return
        self._build_processing_state()
        self._plot_ppg_preview()

    def _plot_ppg_preview(self):
        ppg_vis_filtered = self.ppg_info.get("ppg_vis_filtered")
        if ppg_vis_filtered is None or len(ppg_vis_filtered) == 0:
            ax = self.ppg_preview_plot.canvas.clear_and_get_axes()
            ax.text(0.5, 0.5, "No PPG data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.ppg_preview_plot.draw()
            return
        ppg_time_s = self.ppg_info.get("ppg_time_s", np.array([]))
        ax = self.ppg_preview_plot.canvas.clear_and_get_axes()
        ax.plot(ppg_time_s, ppg_vis_filtered, color="#ff4757", linewidth=1.0)
        ax.set_title("Processed PPG and Detected Peaks")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PPG")
        ax.grid(True, alpha=0.25)
        if self.beat_source_combo.currentText() == "Detect from PPG raw":
            plot_peaks_idx = self.ppg_info.get("ppg_vis_peaks_idx", np.array([], dtype=int))
        else:
            plot_peaks_idx = _map_times_to_indices(ppg_time_s, self.beat_times_s)
        if len(plot_peaks_idx) > 0:
            ax.scatter(ppg_time_s[plot_peaks_idx], ppg_vis_filtered[plot_peaks_idx], s=22, color="#00e5ff", edgecolors="none")
        self.ppg_preview_plot.draw()

    def _start_worker(self, mode: str, inputs: dict):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Analysis is already running.")
            return
        self.worker = AnalysisWorker(mode, inputs)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self._set_status)
        self.worker.failed.connect(self._worker_failed)
        self.worker.finished_result.connect(self._worker_finished)
        self.progress_bar.setValue(0)
        self.worker.start()

    def _worker_failed(self, traceback_text: str):
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis failed")
        self._log(traceback_text)
        QMessageBox.critical(self, "Analysis failed", traceback_text)

    def _worker_finished(self, result: dict):
        self.progress_bar.setValue(100)
        if result.get("warning"):
            self.status_label.setText(result["warning"])
            QMessageBox.warning(self, "Analysis warning", result["warning"])
            return
        if self.worker.mode == "window":
            self.window_result = result
            self._render_window_result(result)
            self.tabs.setCurrentWidget(self.window_tab)
            self.status_label.setText("Window analysis complete")
            return
        self.full_result = result
        self._render_full_result(result)
        self.tabs.setCurrentWidget(self.full_tab)
        self.export_ml_button.setEnabled(True)
        self.export_ml_status.setText("Ready to export")
        for message in result.get("save_messages", []):
            self._log(message)
            QMessageBox.information(self, "Saved", message)
        self.status_label.setText("Full record analysis complete")

    def _window_inputs(self):
        if self.scg_df.empty or len(self.scg_proc_full) == 0:
            return None
        return {
            **self._base_params(),
            "scg_proc_full": self.scg_proc_full,
            "scg_raw": self.scg_raw,
            "fs_infer": self.fs_infer,
            "start_time": self.window_start_spin.value(),
            "window_size": self.window_size_spin.value(),
            "svmd_alpha": self.svmd_alpha_spin.value(),
            "prominence_factor": self.prominence_spin.value(),
            "power_exp": self.power_spin.value(),
            "beat_times_s": self.ppg_info.get("beat_times_s", np.array([], dtype=float)),
            "ppg_peaks_ref": self.ppg_info.get("ppg_peaks_ref", np.array([], dtype=int)),
            "ppg_peaks_full": self.ppg_info.get("ppg_peaks_full", np.array([], dtype=int)),
            "ref_fs": self.ppg_info.get("ref_fs", 100.0),
            "use_iqr_filter": self.use_iqr_filter_check.isChecked(),
            "save_json_output": self.save_json_check.isChecked(),
            "output_folder": self.output_folder_edit.text().strip() or "Saved_Peaks",
            "file_label": self.file_label,
        }

    def _full_inputs(self):
        if self.scg_df.empty or len(self.scg_proc_full) == 0:
            return None
        trim_start = self.trim_start_spin.value()
        trim_end = self.trim_end_spin.value()
        if trim_start >= trim_end:
            QMessageBox.warning(self, "Invalid trim range", "Trim start must be strictly less than trim end.")
            return None
        return {
            **self._base_params(),
            "scg_proc_full": self.scg_proc_full,
            "scg_raw": self.scg_raw,
            "fs_proc": self.fs_proc,
            "fs_infer": self.fs_infer,
            "trim_start": trim_start,
            "trim_end": trim_end,
            "beat_times_s": self.ppg_info.get("beat_times_s", np.array([], dtype=float)),
            "ppg_peaks_ref": self.ppg_info.get("ppg_peaks_ref", np.array([], dtype=int)),
            "ppg_peaks_full": self.ppg_info.get("ppg_peaks_full", np.array([], dtype=int)),
            "ppg_peak_times_s": self.ppg_info.get("ppg_peak_times_s", np.array([], dtype=float)),
            "ref_fs": self.ppg_info.get("ref_fs", 100.0),
            "show_sqa_overlay": self.show_sqa_overlay.isChecked(),
            "sqa_segment_seconds": self.sqa_segment_spin.value(),
            "min_flags_to_reject": self.min_flags_spin.value(),
            "kurt_thresh": self.kurt_thresh_spin.value(),
            "zcr_low": self.zcr_low_spin.value(),
            "zcr_high": self.zcr_high_spin.value(),
            "env_thresh": self.env_thresh_spin.value(),
            "rms_low_percentile": self.rms_low_percentile_spin.value(),
            "rms_high_percentile": self.rms_high_percentile_spin.value(),
            "rms_low_mad_mult": self.rms_low_mad_spin.value(),
            "rms_high_mad_mult": self.rms_high_mad_spin.value(),
            "exclude_bad_windows": self.exclude_bad_windows_check.isChecked(),
            "bad_window_fraction_threshold": self.bad_window_fraction_spin.value(),
            "use_iqr_filter": self.use_iqr_filter_check.isChecked(),
            "svmd_alpha": self.svmd_alpha_spin.value(),
            "prominence_factor": self.prominence_spin.value(),
            "power_exp": self.power_spin.value(),
            "save_json_output": self.save_json_check.isChecked(),
            "save_ppg_json_output": self.save_ppg_json_check.isChecked(),
            "output_folder": self.output_folder_edit.text().strip() or "Saved_Peaks",
            "beat_source": self.beat_source_combo.currentText(),
            "file_label": self.file_label,
        }

    def _run_window_analysis(self):
        if self.scg_df.empty:
            QMessageBox.information(self, "No data", "Load a CSV first.")
            return
        self._build_processing_state()
        inputs = self._window_inputs()
        if inputs is None:
            return
        self._set_status("Running window analysis...")
        self._start_worker("window", inputs)

    def _run_full_analysis(self):
        if self.scg_df.empty:
            QMessageBox.information(self, "No data", "Load a CSV first.")
            return
        self._build_processing_state()
        inputs = self._full_inputs()
        if inputs is None:
            return
        self._set_status("Running full record analysis...")
        self._start_worker("full", inputs)

    def _build_export_defaults(self) -> dict:
        defaults = {}
        meta = self.meta_data or {}
        if meta:
            defaults.update({
                "patient_initials": meta.get("patient_initials", ""),
                "age": meta.get("age"),
                "sex": meta.get("sex"),
                "weight_kg": meta.get("weight_kg"),
                "height_cm": meta.get("height_cm"),
                "bmi": meta.get("bmi"),
                "cardiac_conditions": meta.get("cardiac_conditions", []),
                "notes": meta.get("notes", ""),
                "session_start": meta.get("session_start"),
                "sample_rate_ppg_hz": meta.get("sample_rate_ppg_hz", PPG_SAMPLE_RATE),
                "filter_enabled": meta.get("filter_enabled"),
                "notch_50hz_enabled": meta.get("notch_50hz_enabled"),
            })

        if not defaults and self._subject:
            defaults.update({
                "patient_initials": self._subject.get("initials", ""),
                "age": self._subject.get("age"),
                "sex": self._subject.get("sex"),
                "weight_kg": self._subject.get("weight_kg"),
                "height_cm": self._subject.get("height_cm"),
                "bmi": self._subject.get("bmi"),
                "cardiac_conditions": self._subject.get("conditions", []),
                "notes": self._subject.get("notes", ""),
                "sample_rate_ppg_hz": PPG_SAMPLE_RATE,
            })

        if not defaults.get("patient_initials") and self.file_label:
            defaults["patient_initials"] = self.file_label.split("_")[0]

        return defaults

    def _export_full_record_for_ml(self):
        if self.scg_df.empty:
            QMessageBox.information(self, "No data", "Load a CSV first.")
            return
        if self.full_result is None:
            QMessageBox.information(self, "No analysis", "Run full record analysis before exporting.")
            return

        trim_start = self.trim_start_spin.value()
        trim_end = self.trim_end_spin.value()
        scg_df = self.scg_df.copy()
        if "time_s" not in scg_df.columns:
            t0 = float(scg_df["timestamp_ms"].iloc[0])
            scg_df["time_s"] = (scg_df["timestamp_ms"] - t0) / 1000.0

        if trim_end > trim_start:
            scg_df = scg_df[(scg_df["time_s"] >= trim_start) & (scg_df["time_s"] <= trim_end)].copy()

        if scg_df.empty:
            QMessageBox.warning(self, "No data", "Trim range contains no SCG samples.")
            return

        defaults = self._build_export_defaults()
        dlg = ExportMlDialog(self, patient_id=self.file_label, existing=defaults)
        if dlg.exec_() != QDialog.Accepted:
            return

        meta_fields = dlg.get_metadata()
        meta_fields["patient_id"] = self.file_label
        meta_fields["session_start"] = defaults.get("session_start") or datetime.now().isoformat()
        meta_fields["sample_rate_ppg_hz"] = defaults.get("sample_rate_ppg_hz", PPG_SAMPLE_RATE)
        meta_fields["filter_enabled"] = defaults.get("filter_enabled")
        meta_fields["notch_50hz_enabled"] = defaults.get("notch_50hz_enabled")

        ao_peaks = self.full_result.get("all_ao_peaks", np.array([], dtype=int))
        ppg_peak_times_s = None
        ppg_raw_series = None
        ppg_fs = None
        ref_fs = float(self.ppg_info.get("ref_fs", 0.0))
        ppg_peaks_ref_trim = self.full_result.get("ppg_peaks_ref_trim", np.array([], dtype=int))
        if ref_fs > 0 and len(ppg_peaks_ref_trim) > 0:
            ppg_times = ppg_peaks_ref_trim.astype(float) / ref_fs
            ppg_times = ppg_times - float(trim_start)
            ppg_peak_times_s = ppg_times[ppg_times >= 0.0]

        if not self.ppg_df.empty:
            ppg_df = self.ppg_df.copy()
            if "time_s" not in ppg_df.columns:
                ppg_df["time_s"] = (ppg_df["timestamp_ms"] - float(ppg_df["timestamp_ms"].iloc[0])) / 1000.0
            if trim_end > trim_start:
                ppg_df = ppg_df[(ppg_df["time_s"] >= trim_start) & (ppg_df["time_s"] <= trim_end)].copy()
            if not ppg_df.empty:
                ppg_raw_series = ppg_df[OPTIONAL_PPG_COL].to_numpy(dtype=float)
                _, ppg_fs = _compute_rate_from_ts(ppg_df["timestamp_ms"].to_numpy(dtype=float))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base = os.path.join(script_dir, ML_EXPORT_DIR)
        try:
            exported = export_subject_for_ml(
                scg_df,
                meta_fields,
                ao_peaks,
                self.fs_proc,
                output_base,
                ppg_peak_times_s=ppg_peak_times_s,
                ppg_raw=ppg_raw_series,
                ppg_fs=ppg_fs,
                ppg_key="PPG_Peaks",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        rel_csv = os.path.relpath(exported["csv_path"], script_dir)
        self.export_ml_status.setText(f"Exported: {rel_csv}")
        if hasattr(self, "batch_export_status"):
            self.batch_export_status.setText("")
        QMessageBox.information(
            self,
            "Export complete",
            f"Exported CSV:\n{exported['csv_path']}\n\nMeta JSON:\n{exported['meta_path']}"
        )

    def _batch_export_for_ml(self):
        search_dir = Path(self.search_dir_edit.text().strip() or DEFAULT_SEARCH_DIR)
        if not search_dir.exists() or not search_dir.is_dir():
            QMessageBox.warning(self, "Missing folder", "Workspace folder does not exist.")
            return

        min_duration = float(self.batch_min_seconds_spin.value())
        candidates = sorted(search_dir.rglob("*.csv"))
        if not candidates:
            QMessageBox.information(self, "No files", "No CSV files found for batch export.")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base = os.path.join(script_dir, ML_EXPORT_DIR)

        exported_paths = []
        skipped_paths = []
        failed_paths = []

        def _load_peaks_seconds(json_path: str, key_hint: str) -> np.ndarray:
            if not os.path.exists(json_path):
                return np.array([], dtype=float)
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle) or {}
            times = payload.get(key_hint)
            if times is None and payload:
                times = next(iter(payload.values()))
            if not times:
                return np.array([], dtype=float)
            return np.asarray([parse_timestamp_to_seconds(t) for t in times], dtype=float)

        for path in candidates:
            duration_s = _get_csv_duration(path)
            if duration_s < min_duration:
                skipped_paths.append(path)
                continue

            try:
                raw_df = _read_csv_from_source(None, str(path))
                df = _prepare_df(raw_df)
            except Exception as exc:
                failed_paths.append((path, str(exc)))
                continue

            scg_df = df[df[["x_g", "y_g", "z_g"]].notna().any(axis=1)].copy()
            if scg_df.empty:
                failed_paths.append((path, "No SCG samples found"))
                continue

            ppg_df = df[df[OPTIONAL_PPG_COL].notna()].copy() if OPTIONAL_PPG_COL in df.columns else pd.DataFrame()
            scg_start_ts = float(scg_df["timestamp_ms"].iloc[0])
            scg_end_ts = float(scg_df["timestamp_ms"].iloc[-1])
            if not ppg_df.empty:
                ppg_df = ppg_df[(ppg_df["timestamp_ms"] >= scg_start_ts) & (ppg_df["timestamp_ms"] <= scg_end_ts)].copy()

            meta, _ = _read_metadata_for_csv(str(path))
            meta = meta or {}

            patient_id = path.stem
            meta_fields = {
                "patient_id": patient_id,
                "patient_initials": meta.get("patient_initials") or patient_id.split("_")[0],
                "age": meta.get("age"),
                "sex": meta.get("sex"),
                "weight_kg": meta.get("weight_kg"),
                "height_cm": meta.get("height_cm"),
                "bmi": meta.get("bmi"),
                "cardiac_conditions": meta.get("cardiac_conditions") or ["Normal"],
                "notes": meta.get("notes", ""),
                "session_start": meta.get("session_start") or datetime.now().isoformat(),
                "sample_rate_ppg_hz": meta.get("sample_rate_ppg_hz", PPG_SAMPLE_RATE),
                "filter_enabled": meta.get("filter_enabled"),
                "notch_50hz_enabled": meta.get("notch_50hz_enabled"),
            }

            fs_hint = meta.get("sample_rate_scg_hz") or 0
            if not fs_hint:
                _, inferred_fs = _compute_rate_from_ts(scg_df["timestamp_ms"].to_numpy(dtype=float))
                fs_hint = inferred_fs if inferred_fs > 0 else ML_TARGET_FS

            ppg_peak_times_s = np.array([], dtype=float)
            ppg_raw_series = None
            ppg_fs = None
            if not ppg_df.empty:
                ppg_info = _current_ppg_info(
                    ppg_df,
                    beat_times_s=np.array([], dtype=float),
                    beat_source="Detect from PPG raw",
                    ppg_bp_low=self.ppg_bp_low_spin.value(),
                    ppg_bp_high=self.ppg_bp_high_spin.value(),
                    ppg_max_bpm=self.ppg_max_bpm_spin.value(),
                    ppg_prom=self.ppg_prom_spin.value(),
                    fs_proc=float(fs_hint),
                )
                ppg_peak_times_s = ppg_info.get("ppg_peak_times_s", np.array([], dtype=float))
                ppg_ts0 = float(ppg_df["timestamp_ms"].iloc[0])
                offset_s = (ppg_ts0 - scg_start_ts) / 1000.0
                if ppg_peak_times_s.size > 0:
                    ppg_peak_times_s = ppg_peak_times_s + offset_s

                ppg_raw_series = ppg_df[OPTIONAL_PPG_COL].to_numpy(dtype=float)
                _, ppg_fs = _compute_rate_from_ts(ppg_df["timestamp_ms"].to_numpy(dtype=float))

                ao_json_path = os.path.join("Saved_Peaks", f"{patient_id}_AO_Peaks.json")
                ao_key = f"{patient_id}_AO_Peaks"
                ao_seconds = _load_peaks_seconds(ao_json_path, ao_key)
                if ao_seconds.size > 1 and ppg_peak_times_s.size > 1:
                    ao_idx = np.round(ao_seconds * float(fs_hint)).astype(int)
                    ref_fs = float(ppg_info.get("ref_fs", 0.0))
                    ppg_idx = np.round(ppg_peak_times_s * ref_fs).astype(int) if ref_fs > 0 else np.array([], dtype=int)
                    if ppg_idx.size > 1 and ref_fs > 0:
                        ptt_seconds, _seg = _calculate_high_corr_ptt(ao_idx, ppg_idx, float(fs_hint), ref_fs)
                        if ptt_seconds > 0:
                            ppg_peak_times_s = ppg_peak_times_s - float(ptt_seconds)
                            ppg_peak_times_s = ppg_peak_times_s[ppg_peak_times_s >= 0.0]

            try:
                exported = export_subject_for_ml(
                    scg_df,
                    meta_fields,
                    np.array([], dtype=int),
                    fs_hint,
                    output_base,
                    ppg_peak_times_s=ppg_peak_times_s,
                    ppg_raw=ppg_raw_series,
                    ppg_fs=ppg_fs,
                    ppg_key="PPG_Peaks",
                )
                exported_paths.append(exported["csv_path"])
            except Exception as exc:
                failed_paths.append((path, str(exc)))

        if hasattr(self, "batch_export_status"):
            self.batch_export_status.setText(f"Batch exported: {len(exported_paths)}")
        self.export_ml_status.setText("")

        summary = (
            f"Exported: {len(exported_paths)}\n"
            f"Skipped (duration < {min_duration:.1f}s): {len(skipped_paths)}\n"
            f"Failed: {len(failed_paths)}"
        )
        QMessageBox.information(self, "Batch export complete", summary)

    def _render_batch_recap(self, recap_rows: list[dict], summary: dict) -> None:
        if not hasattr(self, "batch_recap_view"):
            return

        if not recap_rows:
            self.batch_recap_view.setHtml(
                f"<b>No batch recap generated.</b><br>{summary.get('message', '')}"
            )
            return

        header_cells = "".join(
            f"<th style='padding:6px;border-bottom:1px solid #e2e8f0;text-align:left;'>{h}</th>"
            for h in summary.get("headers", [])
        )
        body_rows = ""
        for row in recap_rows:
            cells = "".join(
                f"<td style='padding:6px;border-bottom:1px solid #edf2f7;'>{row.get(h, '')}</td>"
                for h in summary.get("headers", [])
            )
            body_rows += f"<tr>{cells}</tr>"

        skipped = summary.get("skipped", 0)
        failed = summary.get("failed", 0)
        min_duration = summary.get("min_duration", 0.0)
        csv_path = summary.get("csv_path", "")
        csv_line = f"Recap CSV: {csv_path}" if csv_path else "Recap CSV: not saved"

        failed_list = ""
        if summary.get("failed_details"):
            failed_items = "".join(
                f"<li>{item}</li>" for item in summary["failed_details"]
            )
            failed_list = f"<div style='margin-top:6px;'><b>Failed:</b><ul>{failed_items}</ul></div>"

        html = f"""
        <html>
        <body style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size:12px;">
            <div style="margin-bottom:8px;">
                <b>Batch Full Analysis Recap</b><br>
                Processed: {len(recap_rows)} | Skipped (< {min_duration:.1f}s): {skipped} | Failed: {failed}<br>
                {csv_line}
            </div>
            <table style="border-collapse:collapse;width:100%;">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{body_rows}</tbody>
            </table>
            {failed_list}
        </body>
        </html>
        """
        self.batch_recap_view.setHtml(html)

    def _batch_full_record_analysis(self):
        search_dir = Path(self.search_dir_edit.text().strip() or DEFAULT_SEARCH_DIR)
        if not search_dir.exists() or not search_dir.is_dir():
            QMessageBox.warning(self, "Missing folder", "Workspace folder does not exist.")
            return

        def _metric_value(val, digits=4):
            if val is None:
                return "N/A"
            try:
                if np.isfinite(val):
                    return round(float(val), digits)
            except Exception:
                return "N/A"
            return "N/A"

        min_duration = float(self.batch_min_seconds_spin.value())
        candidates = sorted(search_dir.rglob("*.csv"))
        if not candidates:
            QMessageBox.information(self, "No files", "No CSV files found for batch analysis.")
            return

        recap_rows = []
        skipped_paths = []
        failed_paths = []

        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "Batch_Reports")

        for path in candidates:
            duration_s = _get_csv_duration(path)
            if duration_s < min_duration:
                skipped_paths.append(path)
                continue

            try:
                raw_df = _read_csv_from_source(None, str(path))
                df = _prepare_df(raw_df)
            except Exception as exc:
                failed_paths.append((path, str(exc)))
                continue

            scg_df = df[df[["x_g", "y_g", "z_g"]].notna().any(axis=1)].copy()
            if scg_df.empty:
                failed_paths.append((path, "No SCG samples found"))
                continue

            beats_df = df[df["beat_event"] == 1].copy()
            ppg_df = df[df[OPTIONAL_PPG_COL].notna()].copy() if OPTIONAL_PPG_COL in df.columns else pd.DataFrame()

            t0 = float(scg_df["timestamp_ms"].iloc[0])
            scg_df["time_s"] = (scg_df["timestamp_ms"] - t0) / 1000.0
            beats_df["time_s"] = (beats_df["timestamp_ms"] - t0) / 1000.0 if not beats_df.empty else np.array([])
            if not ppg_df.empty:
                ppg_df["time_s"] = (ppg_df["timestamp_ms"] - t0) / 1000.0

            duration_s, actual_hz = _compute_rate(scg_df)
            fs_infer = self.override_hz_spin.value() if self.override_fs_check.isChecked() else actual_hz
            if fs_infer <= 0:
                failed_paths.append((path, "Invalid sampling rate inferred"))
                continue

            scg_raw = scg_df["z_g"].to_numpy(dtype=float)
            scg_proc_full, fs_proc = _processing_signal(
                scg_raw,
                fs_infer,
                float(self.target_fs_spin.value()),
                self.preprocessing_mode.currentText(),
            )

            beat_times_s = beats_df["time_s"].to_numpy(dtype=float) if len(beats_df) > 0 else np.array([], dtype=float)
            ppg_info = _current_ppg_info(
                ppg_df,
                beat_times_s,
                self.beat_source_combo.currentText(),
                self.ppg_bp_low_spin.value(),
                self.ppg_bp_high_spin.value(),
                self.ppg_max_bpm_spin.value(),
                self.ppg_prom_spin.value(),
                fs_proc,
            )

            max_t = float(scg_df["time_s"].iloc[-1]) if not scg_df.empty else 0.0
            if max_t <= 0:
                failed_paths.append((path, "Record duration invalid"))
                continue

            inputs = self._base_params()
            inputs.update({
                "scg_proc_full": scg_proc_full,
                "scg_raw": scg_raw,
                "fs_proc": fs_proc,
                "fs_infer": fs_infer,
                "trim_start": 0.0,
                "trim_end": max_t,
                "beat_times_s": ppg_info.get("beat_times_s", np.array([], dtype=float)),
                "ppg_peaks_ref": ppg_info.get("ppg_peaks_ref", np.array([], dtype=int)),
                "ppg_peaks_full": ppg_info.get("ppg_peaks_full", np.array([], dtype=int)),
                "ppg_peak_times_s": ppg_info.get("ppg_peak_times_s", np.array([], dtype=float)),
                "ref_fs": ppg_info.get("ref_fs", 100.0),
                "show_sqa_overlay": self.show_sqa_overlay.isChecked(),
                "sqa_segment_seconds": self.sqa_segment_spin.value(),
                "min_flags_to_reject": self.min_flags_spin.value(),
                "kurt_thresh": self.kurt_thresh_spin.value(),
                "zcr_low": self.zcr_low_spin.value(),
                "zcr_high": self.zcr_high_spin.value(),
                "env_thresh": self.env_thresh_spin.value(),
                "rms_low_percentile": self.rms_low_percentile_spin.value(),
                "rms_high_percentile": self.rms_high_percentile_spin.value(),
                "rms_low_mad_mult": self.rms_low_mad_spin.value(),
                "rms_high_mad_mult": self.rms_high_mad_spin.value(),
                "exclude_bad_windows": self.exclude_bad_windows_check.isChecked(),
                "bad_window_fraction_threshold": self.bad_window_fraction_spin.value(),
                "use_iqr_filter": self.use_iqr_filter_check.isChecked(),
                "svmd_alpha": self.svmd_alpha_spin.value(),
                "prominence_factor": self.prominence_spin.value(),
                "power_exp": self.power_spin.value(),
                "save_json_output": self.save_json_check.isChecked(),
                "save_ppg_json_output": self.save_ppg_json_check.isChecked(),
                "output_folder": self.output_folder_edit.text().strip() or "Saved_Peaks",
                "beat_source": self.beat_source_combo.currentText(),
                "file_label": path.stem,
            })

            try:
                result = run_full_record_analysis(inputs)
            except Exception as exc:
                failed_paths.append((path, str(exc)))
                continue

            meta, _ = _read_metadata_for_csv(str(path))
            meta = meta or {}
            initials = (meta.get("patient_initials") or path.stem.split("_")[0]).upper()
            sex = meta.get("sex", "")
            bmi = meta.get("bmi", "")
            conds = meta.get("cardiac_conditions") or ["Normal"]
            if not isinstance(conds, list):
                conds = [str(conds)]
            conditions = "; ".join([str(c) for c in conds])

            ptt_best = result.get("ptt_best_segment")
            ptt_segment_corr = "N/A"
            if ptt_best and isinstance(ptt_best, dict):
                corr_val = ptt_best.get("corr")
                ptt_segment_corr = _metric_value(corr_val)

            sqa_rejected = "N/A"
            sqa_result = result.get("sqa_result_full_record")
            if sqa_result and "bad_mask" in sqa_result:
                sqa_rejected = int(np.sum(sqa_result.get("bad_mask", [])))

            recap_rows.append({
                "record": path.name,
                "initials": initials,
                "sex": sex,
                "conditions": conditions,
                "bmi": bmi,
                "duration_s": round(max_t, 2),
                "ao_peaks": int(len(result.get("all_ao_peaks", []))),
                "ppg_peaks": int(len(result.get("ppg_peaks_full_trim", []))),
                "interval_corr": _metric_value(result.get("correlation", np.nan)),
                "ptt_shift_s": _metric_value(result.get("ptt_seconds", np.nan), digits=5),
                "ptt_segment_corr": ptt_segment_corr,
                "sqa_rejected_segments": sqa_rejected,
                "sqa_skipped_windows": result.get("skipped_bad_windows", 0),
            })

        recap_csv_path = ""
        if self.batch_save_csv_check.isChecked():
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recap_csv_path = os.path.join(out_dir, f"batch_full_recap_{timestamp}.csv")
        headers = [
            "record",
            "initials",
            "sex",
            "conditions",
            "bmi",
            "duration_s",
            "ao_peaks",
            "ppg_peaks",
            "interval_corr",
            "ptt_shift_s",
            "ptt_segment_corr",
            "sqa_rejected_segments",
            "sqa_skipped_windows",
        ]

        if recap_rows and recap_csv_path:
            with open(recap_csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                writer.writerows(recap_rows)

        failed_details = [f"{p.name}: {err}" for p, err in failed_paths]
        summary = {
            "skipped": len(skipped_paths),
            "failed": len(failed_paths),
            "min_duration": min_duration,
            "csv_path": recap_csv_path if recap_rows else "",
            "headers": headers,
            "failed_details": failed_details,
            "message": "No eligible records to analyze.",
        }

        self._render_batch_recap(recap_rows, summary)
        if hasattr(self, "batch_full_status"):
            status_text = f"Batch full analysis: {len(recap_rows)}"
            if recap_rows:
                if recap_csv_path:
                    status_text += f" | Recap: {os.path.relpath(recap_csv_path, script_dir)}"
                else:
                    status_text += " | Recap: not saved"
            self.batch_full_status.setText(status_text)

        summary_text = (
            f"Processed: {len(recap_rows)}\n"
            f"Skipped (duration < {min_duration:.1f}s): {len(skipped_paths)}\n"
            f"Failed: {len(failed_paths)}"
        )
        QMessageBox.information(self, "Batch full analysis complete", summary_text)

    def _render_window_result(self, result: dict):
        self.window_raw_plot.clear()
        self.window_filtered_plot.clear()
        self.window_decomposed_plot.clear()
        self.window_reconstructed_plot.clear()
        self.window_power7_plot.clear()
        self.window_detected_power7_plot.clear()
        self.window_peaks_original_plot.clear()

        time_axis = result.get("time_axis", np.array([]))
        scg_window = result.get("scg_window", np.array([]))
        scg_raw_window_resampled = result.get("scg_raw_window_resampled", np.array([]))
        modes = result.get("modes", np.array([]))
        selected_idx = result.get("selected_idx", np.array([], dtype=int))
        center_freq_hz = result.get("center_freq_hz", np.array([]))
        s_ao = result.get("s_ao", np.array([]))
        s_ao_7 = result.get("s_ao_7", np.array([]))
        envelope = result.get("envelope", np.array([]))
        smoothed_env = result.get("smoothed_env", np.array([]))
        peaks = result.get("peaks", np.array([], dtype=int))
        ppg_peaks_window = result.get("ppg_peaks_window", np.array([], dtype=int))

        # 1. Raw signal
        ax = self.window_raw_plot.axes
        if len(scg_raw_window_resampled) > 0:
            ax.plot(time_axis, scg_raw_window_resampled, color="#7f8c8d", linewidth=1.2, label="Raw SCG (z-axis)")
            ax.legend(loc="upper right")
        else:
            ax.text(0.5, 0.5, "Raw signal not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        self.window_raw_plot.draw()

        # 2. Filtered signal
        ax = self.window_filtered_plot.axes
        ax.plot(time_axis, scg_window, color="#1f3a93", linewidth=1.3, label="Filtered SCG")
        ax.legend(loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        self.window_filtered_plot.draw()

        # 3. Decomposed SVMD Modes
        ax = self.window_decomposed_plot.axes
        if len(modes) > 0:
            max_amp = np.max(np.abs(modes))
            spacing = max_amp * 1.5 if max_amp > 0 else 1.0
            for idx, mode in enumerate(modes):
                offset = -idx * spacing
                is_selected = idx in selected_idx
                freq = center_freq_hz[idx] if idx < len(center_freq_hz) else 0.0
                color = "#e74c3c" if is_selected else "#7f8c8d"
                label = f"Mode {idx} ({freq:.2f} Hz) [Selected]" if is_selected else f"Mode {idx} ({freq:.2f} Hz)"
                ax.plot(time_axis, mode + offset, color=color, linewidth=1.1, label=label)
                ax.axhline(offset, color="gray", linestyle=":", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            selected_str = ", ".join([f"Mode {i} ({center_freq_hz[i]:.1f} Hz)" for i in selected_idx])
            ax.set_title(f"3. SVMD Decomposed Modes (Selected: {selected_str if selected_str else 'None'})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (offset)")
        ax.grid(True, alpha=0.25)
        self.window_decomposed_plot.draw()

        # 4. Reconstructed AO Signal
        ax = self.window_reconstructed_plot.axes
        ax.plot(time_axis, s_ao, color="#2c3e50", linewidth=1.2, label="Reconstructed AO Signal")
        ax.legend(loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        self.window_reconstructed_plot.draw()

        # 5. Reconstructed Signal after 7th Power
        ax = self.window_power7_plot.axes
        power_exp = int(self.power_spin.value())
        ax.plot(time_axis, s_ao_7, color="#8e44ad", linewidth=1.2, label=f"AO Signal ({power_exp}th Power)")
        ax.legend(loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        self.window_power7_plot.draw()

        # 6. Detected Peaks at 7th Power (Envelope)
        ax = self.window_detected_power7_plot.axes
        ax.plot(time_axis, s_ao_7, color="#ddd", linewidth=1.0, label="7th Power Signal", alpha=0.7)
        ax.plot(time_axis, envelope, color="#2ecc71", linewidth=1.1, label="Hilbert Envelope")
        ax.plot(time_axis, smoothed_env, color="#f39c12", linewidth=1.3, label="Smoothed Envelope")
        if len(peaks) > 0:
            ax.scatter(time_axis[peaks], smoothed_env[peaks], s=35, color="red", zorder=5, label="Detected Peaks")
        ax.legend(loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        self.window_detected_power7_plot.draw()

        # 7. Original SCG Signal with Detected Peaks
        ax = self.window_peaks_original_plot.axes
        if len(scg_raw_window_resampled) > 0:
            ax.plot(time_axis, scg_raw_window_resampled, color="#7f8c8d", linewidth=1.2, label="Raw SCG")
            if len(peaks) > 0:
                ax.scatter(time_axis[peaks], scg_raw_window_resampled[peaks], s=35, color="red", zorder=5, label="AO Peaks on Raw")
        else:
            ax.plot(time_axis, scg_window, color="#1f3a93", linewidth=1.3, label="Filtered SCG")
            if len(peaks) > 0:
                ax.scatter(time_axis[peaks], scg_window[peaks], s=35, color="red", zorder=5, label="AO Peaks on Filtered")
        
        # Also overlay PPG vertical lines as reference like before
        if len(ppg_peaks_window) > 0:
            for bt in time_axis[ppg_peaks_window]:
                ax.axvline(float(bt), color="#27ae60", linestyle="--", linewidth=1)
            # Add PPG dummy line for the legend
            ax.plot([], [], color="#27ae60", linestyle="--", linewidth=1, label="PPG Beat Reference")

        ax.legend(loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        self.window_peaks_original_plot.draw()

        rows = []
        comp = result.get("comparison") or {}
        for key in ["correlation", "rmse", "mae", "mean_diff", "std_diff"]:
            if key in comp:
                rows.append((key, comp[key]))
        detection = result.get("detection_metrics")
        if detection:
            rows.extend([(k, detection.get(k)) for k in ["TP", "FP", "FN", "SE", "P", "ACC", "DER"]])
        self._populate_key_value_table(self.window_metrics_table, rows)

    def _render_full_result(self, result: dict):
        for plot in [self.full_raw_plot, self.full_proc_plot, self.full_ppg_plot, self.full_interval_plot, self.full_ba_plot, self.full_corr_plot, self.full_shift_plot]:
            plot.clear()

        full_time_axis = result.get("full_time_axis", np.array([]))
        scg_raw_display = result.get("scg_raw_display", np.array([]))
        scg_proc_trim = result.get("scg_proc_trim", np.array([]))
        ppg_peaks_full_trim = result.get("ppg_peaks_full_trim", np.array([], dtype=int))
        original_ppg_peaks_full_trim = result.get("original_ppg_peaks_full_trim", np.array([], dtype=int))
        ppg_intervals_full = result.get("ppg_intervals_full", np.array([]))
        ppg_interval_times_full = result.get("ppg_interval_times_full", np.array([]))
        all_ao_peaks = result.get("all_ao_peaks", np.array([], dtype=int))
        all_ao_intervals = result.get("all_ao_intervals", np.array([]))
        all_ao_intervals_times = result.get("all_ao_intervals_times", np.array([]))
        detection_metrics_full = result.get("detection_metrics_full")
        sample_bad_mask_full_record = result.get("sample_bad_mask_full_record")
        ppg_vis_filtered = self.ppg_info.get("ppg_vis_filtered")
        ppg_time_s = self.ppg_info.get("ppg_time_s", np.array([]))
        ppg_vis_peaks_idx = self.ppg_info.get("ppg_vis_peaks_idx", np.array([], dtype=int))
        trim_start = self.trim_start_spin.value()

        ax = self.full_raw_plot.axes
        ax.plot(full_time_axis, scg_raw_display, color="#7f8c8d", linewidth=1.0, label="SCG Raw")
        if len(all_ao_peaks) > 0:
            ax.scatter(full_time_axis[all_ao_peaks], scg_raw_display[all_ao_peaks], s=18, color="red", label="AO Peaks")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        self.full_raw_plot.draw()

        ax = self.full_proc_plot.axes
        if sample_bad_mask_full_record is not None and self.show_sqa_overlay.isChecked():
            scg_good = np.where(~sample_bad_mask_full_record, scg_proc_trim, np.nan)
            scg_bad = np.where(sample_bad_mask_full_record, scg_proc_trim, np.nan)
            ax.plot(full_time_axis, scg_good, color="#1f3a93", linewidth=1.0, label="SCG Good")
            ax.plot(full_time_axis, scg_bad, color="red", linewidth=1.0, label="SCG Bad")
        else:
            ax.plot(full_time_axis, scg_proc_trim, color="#1f3a93", linewidth=1.0, label="SCG")
        if len(all_ao_peaks) > 0:
            ax.scatter(full_time_axis[all_ao_peaks], scg_proc_trim[all_ao_peaks], s=18, color="red", label="AO Peaks")
        if len(ppg_peaks_full_trim) > 0:
            ax.scatter(full_time_axis[ppg_peaks_full_trim], scg_proc_trim[ppg_peaks_full_trim], s=28, color="green", marker="x", label="PPG Beats")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        self.full_proc_plot.draw()

        ax = self.full_ppg_plot.axes
        if ppg_vis_filtered is not None and len(ppg_vis_filtered) > 0:
            ax.plot(ppg_time_s, ppg_vis_filtered, color="#ff4757", linewidth=1.2, label="PPG Filtered")
            peaks_idx = ppg_vis_peaks_idx if self.beat_source_combo.currentText() == "Detect from PPG raw" else _map_times_to_indices(ppg_time_s, self.beat_times_s)
            if len(peaks_idx) > 0:
                ax.scatter(ppg_time_s[peaks_idx], ppg_vis_filtered[peaks_idx], s=20, color="#00e5ff", label="PPG Peaks")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PPG")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        self.full_ppg_plot.draw()

        ax = self.full_interval_plot.axes
        ptt_best_segment = result.get("ptt_best_segment")  # dict with t_start, t_end, corr
        if len(ppg_interval_times_full) > 0:
            ax.plot(ppg_interval_times_full, ppg_intervals_full, color="green", linewidth=1.2, marker="o", markersize=3, label="PPG-PPG")
        if len(all_ao_intervals_times) > 0:
            ax.plot(all_ao_intervals_times, all_ao_intervals, color="red", linewidth=1.2, marker="o", markersize=3, label="AO-AO")
        # Highlight the best-correlation segment used to determine PTT
        if ptt_best_segment is not None:
            seg_t0 = ptt_best_segment["t_start"]   # already in absolute time
            seg_t1 = ptt_best_segment["t_end"]
            seg_corr = ptt_best_segment["corr"]
            ax.axvspan(seg_t0, seg_t1, alpha=0.15, color="#f9ca24", zorder=0, label=f"PTT Segment (r={seg_corr:.3f})")
            ax.axvline(seg_t0, color="#e67e22", linewidth=1.2, linestyle="--", zorder=1)
            ax.axvline(seg_t1, color="#e67e22", linewidth=1.2, linestyle="--", zorder=1)
            ax.text(
                (seg_t0 + seg_t1) / 2, 0.98,
                f"r={seg_corr:.3f}",
                ha="center", va="top", fontsize=8, color="#e67e22", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff9e6", edgecolor="#f9ca24", alpha=0.9),
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Interval (ms)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        self.full_interval_plot.draw()

        if len(result.get("mean_intervals", np.array([]))) > 0:
            mean_intervals = result.get("mean_intervals", np.array([]))
            diff_intervals = result.get("diff_intervals", np.array([]))
            mean_diff = result.get("mean_diff", np.nan)
            std_diff = result.get("std_diff", np.nan)
            ax = self.full_ba_plot.axes
            ax.scatter(mean_intervals, diff_intervals, s=18, color=(0, 0, 1, 0.45), label="Data points")
            if np.isfinite(mean_diff):
                ax.axhline(float(mean_diff), color="red", linewidth=2, label=f"Mean: {mean_diff:.1f} ms")
            if np.isfinite(std_diff):
                upper = mean_diff + 1.96 * std_diff
                lower = mean_diff - 1.96 * std_diff
                ax.axhline(float(upper), color="gray", linestyle="--", linewidth=1.2, label=f"+1.96 SD: {upper:.1f} ms")
                ax.axhline(float(lower), color="gray", linestyle="--", linewidth=1.2, label=f"-1.96 SD: {lower:.1f} ms")
            ax.set_xlabel("Mean of AO-AO and PPG-PPG (ms)")
            ax.set_ylabel("Difference (AO-AO - PPG-PPG) (ms)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            self.full_ba_plot.draw()

            ppg_vals = result.get("ppg_intervals_matched", np.array([]))
            ao_vals = result.get("ao_intervals_matched", np.array([]))
            if len(ppg_vals) > 0 and len(ao_vals) > 0:
                ax = self.full_corr_plot.axes
                ax.scatter(ppg_vals, ao_vals, s=18, color="green", label="Intervals")
                min_val = float(min(np.min(ppg_vals), np.min(ao_vals)))
                max_val = float(max(np.max(ppg_vals), np.max(ao_vals)))
                ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Identity line")
                ax.set_xlabel("PPG-PPG Interval (ms)")
                ax.set_ylabel("AO-AO Interval (ms)")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")
                self.full_corr_plot.draw()

        # Render the PPG peak shift alignment plot (Before vs After)
        ptt_seconds = result.get("ptt_seconds", 0.0)
        ptt_ms = ptt_seconds * 1000.0
        shift_ppg_align = bool(ptt_seconds > 0.0)
        
        ax = self.full_shift_plot.axes
        if len(scg_proc_trim) > 0 and len(full_time_axis) == len(scg_proc_trim):
            # Plot the entire SCG signal in neutral grey
            ax.plot(full_time_axis, scg_proc_trim, color="#7f8c8d", linewidth=1.0, label="SCG Signal")
            
            # Draw original PPG peaks as vertical red lines
            if len(original_ppg_peaks_full_trim) > 0:
                valid_orig_idx = original_ppg_peaks_full_trim[original_ppg_peaks_full_trim < len(scg_proc_trim)]
                _first = True
                for idx in valid_orig_idx:
                    ax.axvline(full_time_axis[idx], color="#ff4757", linewidth=0.8, linestyle="-",
                               alpha=0.7, label="Original PPG Peaks" if _first else "", zorder=2)
                    _first = False
            
            # Draw shifted PPG peaks as vertical green solid lines
            if shift_ppg_align and ptt_seconds > 0.0 and len(ppg_peaks_full_trim) > 0:
                valid_shift_idx = ppg_peaks_full_trim[ppg_peaks_full_trim < len(scg_proc_trim)]
                _first = True
                for idx in valid_shift_idx:
                    ax.axvline(full_time_axis[idx], color="#2ed573", linewidth=0.8, linestyle="-",
                               alpha=0.7, label="Shifted PPG Peaks" if _first else "", zorder=3)
                    _first = False
            
            if shift_ppg_align and ptt_seconds > 0.0:
                ax.set_title(f"PPG Peak Shift Alignment (Shifted by {ptt_ms:.1f} ms)", fontsize=9)
            else:
                ax.set_title("PPG Peak Shift Alignment (No Shift Applied)", fontsize=9)
                    
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SCG Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        self.full_shift_plot.draw()

        self._fill_full_tables(result, detection_metrics_full)

    def _fill_full_tables(self, result: dict, detection_metrics_full: dict):
        paper_metrics = result.get("paper_metrics") or {}
        ptt_metrics = result.get("ptt_metrics") or {}
        interval_rows = []
        
        ptt_seconds = result.get("ptt_seconds", 0.0)
        interval_rows.append(("PTT Shift Alignment (ms)", ptt_seconds * 1000.0))
        
        for key in ["correlation", "rmse", "mae", "mean_diff", "std_diff"]:
            if key in result:
                interval_rows.append((key, result.get(key)))
        for key in ["mean_scg_hr", "mean_ref_hr", "ARE", "AAE", "AAEP", "ba_bias", "ba_upper_loa", "ba_lower_loa"]:
            if key in paper_metrics:
                interval_rows.append((key, paper_metrics.get(key)))
        for key in ["mean_ptt_ms", "std_ptt_ms", "ptt_rr_correlation"]:
            if key in ptt_metrics:
                interval_rows.append((key, ptt_metrics.get(key)))
        self._populate_key_value_table(self.full_metrics_table, interval_rows)

        if detection_metrics_full:
            detect_rows = [(k, detection_metrics_full.get(k)) for k in ["TP", "FP", "FN", "SE", "P", "ACC", "DER"]]
            self._populate_key_value_table(self.detect_metrics_table, detect_rows)
            fp_times = detection_metrics_full.get("fp_times_s", np.array([]))
            fn_times = detection_metrics_full.get("fn_times_s", np.array([]))
            fp_peaks = detection_metrics_full.get("fp_peaks", np.array([]))
            fn_peaks = detection_metrics_full.get("fn_peaks", np.array([]))
            detect_df = pd.DataFrame()
            if len(fp_times) > 0:
                detect_df = pd.concat([detect_df, pd.DataFrame({"type": ["FP"] * len(fp_times), "sample_idx": fp_peaks, "time_s": fp_times})], ignore_index=True)
            if len(fn_times) > 0:
                detect_df = pd.concat([detect_df, pd.DataFrame({"type": ["FN"] * len(fn_times), "sample_idx": fn_peaks, "time_s": fn_times})], ignore_index=True)
            self._fill_table(self.detection_errors_table, detect_df)
        else:
            self.detect_metrics_table.clear()
            self.detection_errors_table.clear()

        ao_removed = result.get("iqr_removed_ao_centers_s", np.array([]))
        ppg_removed = result.get("iqr_removed_ppg_centers_s", np.array([]))
        if len(ao_removed) > 0 or len(ppg_removed) > 0:
            df = pd.concat([
                pd.DataFrame({"type": ["AO"] * len(ao_removed), "center_s": ao_removed + self.trim_start_spin.value()}),
                pd.DataFrame({"type": ["PPG"] * len(ppg_removed), "center_s": ppg_removed + self.trim_start_spin.value()}),
            ], ignore_index=True)
            self._fill_table(self.iqr_table, df)
        else:
            self.iqr_table.clear()

    def _populate_key_value_table(self, table: QTableWidget, rows):
        table.clear()
        table.setRowCount(len(rows))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        for row_index, (key, value) in enumerate(rows):
            table.setItem(row_index, 0, QTableWidgetItem(str(key)))
            if isinstance(value, float):
                text = f"{value:.4f}" if np.isfinite(value) else "nan"
            else:
                text = str(value)
            table.setItem(row_index, 1, QTableWidgetItem(text))
        table.resizeColumnsToContents()

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame):
        _table_from_dataframe(table, df)

    # ═══════════════════════════════════════════════════════════════════════════
    # Capture & Serial Processing Methods
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
        self._ingest_timer.start(20)
        self._rate_timer.start(1000)
        if self._subject is not None:
            self._rec_btn.setEnabled(True)

    def _on_disconnect(self):
        if self._is_recording:
            self._stop_recording()
        if self._thread:
            if self._reader:
                try:
                    self._reader.error.disconnect(self._on_serial_error)
                except TypeError:
                    pass
            self._thread.stop()
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
        if self._thread is None:
            return
        self._status_lbl.setText("ERROR")
        self._status_lbl.setObjectName("status_err")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        QMessageBox.critical(self, "Serial Error", msg)
        self._on_disconnect()

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
        self._notch_cb.setEnabled(checked)
        if not checked:
            self._notch_cb.setChecked(False)
        self._reset_filter_state()
        self._apply_scg_yrange()

    def _on_notch_toggled(self, checked: bool):
        self._notch_on = checked
        self._reset_filter_state()

    def _on_artifact_toggled(self, checked: bool):
        self._artifact_rejection_on = checked
        self._ar_buf_x[:] = 0.0
        self._ar_buf_y[:] = 0.0
        self._ar_buf_z[:] = 0.0
        self._ar_idx  = 0
        self._ar_full = False
        self._ar_prev = None

    def _artifact_reject(self, xg: float, yg: float, zg: float):
        AR_THRESHOLD = 8.0
        MIN_SAMPLES  = self._ar_win // 2

        self._ar_buf_x[self._ar_idx] = xg
        self._ar_buf_y[self._ar_idx] = yg
        self._ar_buf_z[self._ar_idx] = zg
        self._ar_idx = (self._ar_idx + 1) % self._ar_win
        if self._ar_idx == 0:
            self._ar_full = True

        n_valid = self._ar_win if self._ar_full else self._ar_idx
        if n_valid < MIN_SAMPLES or self._ar_prev is None:
            self._ar_prev = (xg, yg, zg)
            return xg, yg, zg

        buf_x = self._ar_buf_x if self._ar_full else self._ar_buf_x[:n_valid]
        buf_y = self._ar_buf_y if self._ar_full else self._ar_buf_y[:n_valid]
        buf_z = self._ar_buf_z if self._ar_full else self._ar_buf_z[:n_valid]

        med_x = np.median(buf_x);  mad_x = np.median(np.abs(buf_x - med_x))
        med_y = np.median(buf_y);  mad_y = np.median(np.abs(buf_y - med_y))
        med_z = np.median(buf_z);  mad_z = np.median(np.abs(buf_z - med_z))

        is_artifact = (
            abs(xg - med_x) > AR_THRESHOLD * max(mad_x, 0.001) or
            abs(yg - med_y) > AR_THRESHOLD * max(mad_y, 0.001) or
            abs(zg - med_z) > AR_THRESHOLD * max(mad_z, 0.001)
        )

        if is_artifact:
            self._ar_rejected_count += 1
            px, py, pz = self._ar_prev
            self._ar_buf_x[(self._ar_idx - 1) % self._ar_win] = px
            self._ar_buf_y[(self._ar_idx - 1) % self._ar_win] = py
            self._ar_buf_z[(self._ar_idx - 1) % self._ar_win] = pz
            return px, py, pz

        self._ar_prev = (xg, yg, zg)
        return xg, yg, zg

    def _apply_scg_yrange(self):
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

    def _drain_serial(self):
        if not self._reader:
            return
        batch = self._reader.drain()
        if batch is None:
            return
        scg_samples, ppg_samples, beat_timestamps, parse_errors = batch

        if len(scg_samples)  > 256: scg_samples  = scg_samples[-256:]
        if len(ppg_samples)  > 100: ppg_samples  = ppg_samples[-100:]
        if len(beat_timestamps) > 20: beat_timestamps = beat_timestamps[-20:]

        self._ingest(scg_samples, ppg_samples, beat_timestamps, parse_errors)

    def _ingest(self, scg_samples, ppg_samples, beat_timestamps, parse_errors):
        self._parse_err_count += int(parse_errors)
        scg_dt  = 1000.0 / SAMPLE_RATE
        ppg_dt  = 1000.0 / PPG_SAMPLE_RATE
        now_ms  = time.monotonic() * 1000.0

        for ts, x_raw, y_raw, z_raw in scg_samples:
            if self._scg_host_clock_ms is None:
                self._scg_host_clock_ms = now_ms
            else:
                self._scg_host_clock_ms += scg_dt

            xg = raw_to_g(x_raw)
            yg = raw_to_g(y_raw)
            zg = raw_to_g(z_raw)
            if self._artifact_rejection_on:
                xg, yg, zg = self._artifact_reject(xg, yg, zg)
            if self._filter_on:
                xg, yg, zg = self._apply_filter(xg, yg, zg)

            hts = int(self._scg_host_clock_ms)
            its = int(ts)

            self._rb_scg_ts.append(its)
            self._rb_scg_hts.append(hts)
            self._rb_scg_x.append(xg)
            self._rb_scg_y.append(yg)
            self._rb_scg_z.append(zg)

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

    def _on_plot_timer(self):
        if not self._plot_dirty:
            return
        self._refresh_scg_plots()
        self._refresh_ppg_plot()
        if self._segment_dirty:
            self._refresh_segment_plot()
            self._segment_dirty = False
        self._plot_dirty = False

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
        t_axis        = (ts_arr - now_ts).astype(np.float32) / 1000.0

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
                for _, line in self._beat_line_cache[i]:
                    pw.removeItem(line)
                self._beat_line_cache[i].clear()
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
                for b_ts, line in self._beat_line_cache[i]:
                    age_s = (now_ts - b_ts) / 1000.0
                    line.setPos(-age_s if 0.0 <= age_s <= WINDOW_SECS
                                else -(WINDOW_SECS + 1))

        if beats_changed:
            self._beat_ts_snapshot = list(beat_src)

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

    def _refresh_stats(self):
        self._stat_beats.setText(str(len(self._beat_ts)))
        self._stat_lost.setText(str(self._parse_err_count))
        self._stat_ar.setText(str(self._ar_rejected_count))
        if len(self._beat_intervals) >= 2:
            bpm = 60000.0 / float(np.mean(self._beat_intervals))
            self._bpm_label.setText(f"{bpm:.0f}")
        else:
            self._bpm_label.setText("--")

    def _update_rate(self):
        self._stat_rate.setText(f"{self._rate_count} Hz")
        self._rate_count = 0

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

        script_dir   = os.path.dirname(os.path.abspath(__file__))
        root_dir     = os.path.join(script_dir, "SUBJECT_Data")
        date_dir     = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)

        ts_str       = now.strftime("%H%M%S")
        default_name = f"{initials}_{ts_str}.csv"
        path         = os.path.join(date_dir, default_name)

        if os.path.exists(path):
            base = os.path.splitext(default_name)[0]
            counter = 1
            while os.path.exists(os.path.join(date_dir, f"{base}_{counter}.csv")):
                counter += 1
            path = os.path.join(date_dir, f"{base}_{counter}.csv")

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
        rel_path = os.path.relpath(path, script_dir)
        self._rec_info_lbl.setText(rel_path)
        self._status_lbl.setText("RECORDING")
        self._status_lbl.setObjectName("status_rec")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        self._rec_timer.start(1000)

        self._write_sidecar_json(path, s, now, cond_str)

    def _stop_recording(self):
        if not self._is_recording:
            return
        self._is_recording = False
        self._rec_timer.stop()

        writer = self._async_writer
        self._async_writer = None

        if writer:
            writer.close()

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
        self._refresh_workspace_files()

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
            pass

    def _tick_rec_timer(self):
        if not self._is_recording:
            return
        self._record_elapsed += 1
        mm = self._record_elapsed // 60
        ss = self._record_elapsed % 60
        self._rec_time_lbl.setText(f"{mm:02d}:{ss:02d}")

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
        self._ar_rejected_count = 0
        self._ar_buf_x[:] = 0.0
        self._ar_buf_y[:] = 0.0
        self._ar_buf_z[:] = 0.0
        self._ar_idx  = 0
        self._ar_full = False
        self._ar_prev = None
        self._beat_ts_snapshot = []
        for cache in self._beat_line_cache:
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

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame):
        _table_from_dataframe(table, df)


def main():
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
    app.setStyleSheet(STYLESHEET)

    window = RawScgSvmdWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
