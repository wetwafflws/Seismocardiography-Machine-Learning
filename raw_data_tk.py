"""
Simplified SCG/PPG monitor using Tkinter.
- Reads STM32 USB CDC packets.
- Draws rolling SCG (X/Y/Z) and PPG plots on Canvas widgets.
- Shows BPM, sample rate, and parse errors.
"""

import struct
import time
import threading
import queue
from collections import deque

import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox

MAGIC = 0xAA
TYPE_SCG = 0x01
TYPE_BEAT = 0x02
TYPE_PPG = 0x03
SCG_PKT_LEN = 13
BEAT_PKT_LEN = 7
PPG_PKT_LEN = 11

WINDOW_SECS = 5
SAMPLE_RATE = 256
PPG_SAMPLE_RATE = 100
WINDOW_N = WINDOW_SECS * SAMPLE_RATE
WINDOW_PPG_N = WINDOW_SECS * PPG_SAMPLE_RATE
BPM_HISTORY = 8

ADC_FULL_SCALE_COUNTS = 65535.0
ADC_ZERO_G_COUNTS = ADC_FULL_SCALE_COUNTS / 2.0
ADC_VREF = 3.3
ADXL335_SENSITIVITY_V_PER_G = 0.3
ADC_COUNTS_PER_G = (ADXL335_SENSITIVITY_V_PER_G / ADC_VREF) * ADC_FULL_SCALE_COUNTS
PPG_MAX_COUNTS = 262143.0


def xor_checksum(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
    return c


def parse_packets(buf: bytearray):
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


def raw_packet_int16_to_adc_counts(raw_value: int) -> float:
    return float(raw_value & 0xFFFF)


def adc_counts_to_g(adc_counts: float, zero_g_counts: float = ADC_ZERO_G_COUNTS) -> float:
    return (adc_counts - zero_g_counts) / ADC_COUNTS_PER_G


def ppg_counts_to_norm(ppg_counts: float) -> float:
    return ppg_counts / PPG_MAX_COUNTS


class SerialWorker(threading.Thread):
    def __init__(self, port: str, baud: int, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self._port = port
        self._baud = baud
        self._out = out_queue
        self._stop = threading.Event()
        self._buf = bytearray()

    def _emit(self, item):
        try:
            self._out.put_nowait(item)
        except queue.Full:
            try:
                self._out.get_nowait()
            except queue.Empty:
                return
            try:
                self._out.put_nowait(item)
            except queue.Full:
                pass

    def run(self):
        try:
            ser = serial.Serial(self._port, self._baud, timeout=0.02)
        except serial.SerialException as e:
            self._emit(("error", str(e)))
            return

        while not self._stop.is_set():
            try:
                chunk = ser.read(256)
                if not chunk:
                    continue
                self._buf.extend(chunk)
                scg, ppg, beats, self._buf, parse_errors = parse_packets(self._buf)
                if scg or ppg or beats or parse_errors:
                    self._emit(("data", scg, ppg, beats, parse_errors))
            except serial.SerialException as e:
                self._emit(("error", str(e)))
                break

        if ser.is_open:
            ser.close()

    def stop(self):
        self._stop.set()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SCG / PPG Monitor (Tk)")
        self.geometry("1100x720")

        self._queue = queue.Queue(maxsize=8)
        self._worker = None

        self._scg_x = deque(maxlen=WINDOW_N)
        self._scg_y = deque(maxlen=WINDOW_N)
        self._scg_z = deque(maxlen=WINDOW_N)
        self._ppg = deque(maxlen=WINDOW_PPG_N)
        self._beat_ts = []
        self._beat_intervals = deque(maxlen=BPM_HISTORY)
        self._last_beat_ts = None

        self._sample_count = 0
        self._parse_errors = 0
        self._rate_count = 0

        self._build_ui()
        self.after(33, self._poll_queue)
        self.after(1000, self._update_rate)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(top, text="Port").pack(side=tk.LEFT)
        self._port_var = tk.StringVar()
        self._port_combo = ttk.Combobox(top, textvariable=self._port_var, width=28)
        self._refresh_ports()
        self._port_combo.pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Refresh", command=self._refresh_ports).pack(side=tk.LEFT, padx=4)
        self._connect_btn = ttk.Button(top, text="Connect", command=self._connect)
        self._connect_btn.pack(side=tk.LEFT, padx=4)
        self._disconnect_btn = ttk.Button(top, text="Disconnect", command=self._disconnect, state=tk.DISABLED)
        self._disconnect_btn.pack(side=tk.LEFT, padx=4)

        stats = ttk.Frame(self)
        stats.pack(fill=tk.X, padx=10, pady=6)
        self._bpm_var = tk.StringVar(value="--")
        self._rate_var = tk.StringVar(value="-- Hz")
        self._err_var = tk.StringVar(value="0")
        ttk.Label(stats, text="BPM:").pack(side=tk.LEFT)
        ttk.Label(stats, textvariable=self._bpm_var, width=6).pack(side=tk.LEFT)
        ttk.Label(stats, text="Rate:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(stats, textvariable=self._rate_var, width=8).pack(side=tk.LEFT)
        ttk.Label(stats, text="Parse errors:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(stats, textvariable=self._err_var, width=8).pack(side=tk.LEFT)
        self._draw_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats, text="Draw", variable=self._draw_enabled).pack(side=tk.LEFT, padx=(12, 0))

        plots = ttk.Frame(self)
        plots.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self._scg_canvas = tk.Canvas(plots, bg="#111522", height=260)
        self._scg_canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Label(plots, text="PPG").pack(anchor=tk.W, pady=(8, 2))
        self._ppg_canvas = tk.Canvas(plots, bg="#111522", height=160)
        self._ppg_canvas.pack(fill=tk.BOTH, expand=True)

    def _refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        values = [p.device for p in ports]
        self._port_combo["values"] = values
        if values:
            self._port_combo.current(0)

    def _connect(self):
        port = self._port_var.get().strip()
        if not port:
            return
        self._worker = SerialWorker(port, 115200, self._queue)
        self._worker.start()
        self._connect_btn.config(state=tk.DISABLED)
        self._disconnect_btn.config(state=tk.NORMAL)

    def _disconnect(self):
        if self._worker:
            self._worker.stop()
            self._worker = None
        self._connect_btn.config(state=tk.NORMAL)
        self._disconnect_btn.config(state=tk.DISABLED)

    def _poll_queue(self):
        try:
            processed = 0
            while processed < 4:
                item = self._queue.get_nowait()
                if not item:
                    break
                if item[0] == "error":
                    messagebox.showerror("Serial Error", item[1])
                    self._disconnect()
                elif item[0] == "data":
                    _, scg, ppg, beats, parse_errors = item
                    self._parse_errors += int(parse_errors)
                    self._err_var.set(str(self._parse_errors))
                    for ts, x, y, z in scg:
                        x_g = adc_counts_to_g(raw_packet_int16_to_adc_counts(x))
                        y_g = adc_counts_to_g(raw_packet_int16_to_adc_counts(y))
                        z_g = adc_counts_to_g(raw_packet_int16_to_adc_counts(z))
                        self._scg_x.append(x_g)
                        self._scg_y.append(y_g)
                        self._scg_z.append(z_g)
                        self._sample_count += 1
                        self._rate_count += 1
                    for ts, ppg_raw in ppg:
                        self._ppg.append(ppg_counts_to_norm(float(ppg_raw)))
                    for ts in beats:
                        if self._last_beat_ts is not None:
                            interval_ms = ts - self._last_beat_ts
                            if 300 < interval_ms < 2000:
                                self._beat_intervals.append(interval_ms)
                        self._last_beat_ts = ts
                    self._update_bpm()
                    if self._draw_enabled.get():
                        self._redraw_plots()
                processed += 1
        except queue.Empty:
            pass

        self.after(33, self._poll_queue)

    def _update_bpm(self):
        if len(self._beat_intervals) >= 2:
            bpm = 60000.0 / (sum(self._beat_intervals) / len(self._beat_intervals))
            self._bpm_var.set(f"{bpm:.0f}")
        else:
            self._bpm_var.set("--")

    def _update_rate(self):
        self._rate_var.set(f"{self._rate_count} Hz")
        self._rate_count = 0
        self.after(1000, self._update_rate)

    def _redraw_plots(self):
        self._draw_series(self._scg_canvas, [self._scg_x, self._scg_y, self._scg_z],
                          colors=["#00e5ff", "#a78bfa", "#2ed573"], y_range=(-2.0, 2.0))
        self._draw_series(self._ppg_canvas, [self._ppg], colors=["#ff4757"], y_range=(0.0, 1.0))

    def _draw_series(self, canvas: tk.Canvas, series_list, colors, y_range):
        w = max(canvas.winfo_width(), 1)
        h = max(canvas.winfo_height(), 1)
        canvas.delete("all")

        y_min, y_max = y_range
        y_span = y_max - y_min if y_max != y_min else 1.0

        for series, color in zip(series_list, colors):
            if len(series) < 2:
                continue
            pts = []
            step = max(1, len(series) // (w - 2))
            trimmed = list(series)[-w * step:]
            for i in range(0, len(trimmed), step):
                x = int(i / step)
                val = trimmed[i]
                y = int(h - ((val - y_min) / y_span) * h)
                pts.append((x, y))
            for i in range(1, len(pts)):
                canvas.create_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1],
                                   fill=color, width=1)


if __name__ == "__main__":
    app = App()
    app.mainloop()
