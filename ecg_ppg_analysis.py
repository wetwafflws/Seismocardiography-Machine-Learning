import sys, csv
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import pearsonr, linregress

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSlider, QSpinBox, QGridLayout,
)
from PyQt5.QtCore import Qt

import pyqtgraph as pg
pg.setConfigOption("background", "#0d1117")
pg.setConfigOption("foreground", "#c9d1d9")
pg.setConfigOptions(antialias=True)

# ─── Constants ───────────────────────────────────────────────────────
FS_ECG = 150.0
FS_PPG = 100.0
MIN_INTERVAL = 0.25
MAX_INTERVAL = 2.50
MIN_WIN = 5.0
MAX_WIN = 60.0
MAX_START = 110.0

# ─── Load full data (no windowing) ────────────────────────────────────
def load_full_data():
    with open("/Users/belugayy/Documents/Campus/Semester 8/ecg_MDSK.csv") as f:
        reader = csv.DictReader(f)
        ecg = np.array([(float(r["time_s"]), float(r["ecg_mv"])) for r in reader])
    ecg_t, ecg_mv = ecg[:, 0], ecg[:, 1]

    with open("/Users/belugayy/Documents/Campus/Semester 8/TA/"
              "Seismocardiography-Machine-Learning/SUBJECT_Data/"
              "2026-06-02/PPG_163718.csv") as f:
        rows = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(rows)
        ppg_ts, ppg_v = [], []
        for r in reader:
            v = r["ppg_raw"]
            if v and v.strip():
                ppg_ts.append(float(r["timestamp_ms"]))
                ppg_v.append(float(v))
    ppg_ts = (np.array(ppg_ts) - ppg_ts[0]) / 1000.0
    ppg_v = np.array(ppg_v)

    return {"ecg_t": ecg_t, "ecg_mv": ecg_mv, "ppg_ts": ppg_ts, "ppg_v": ppg_v}


def window_and_process(full, t_start, t_win, pct):
    # --- Window ECG ---
    m = (full["ecg_t"] >= t_start) & (full["ecg_t"] < t_start + t_win)
    ecg_t = full["ecg_t"][m]
    ecg_mv = full["ecg_mv"][m]

    # R-peaks
    height = np.percentile(ecg_mv, 97) if len(ecg_mv) else 0
    r_peaks, _ = find_peaks(ecg_mv, distance=int(FS_ECG * 0.35), height=height)
    r_times = ecg_t[r_peaks]

    # --- Window PPG ---
    m = (full["ppg_ts"] >= t_start) & (full["ppg_ts"] < t_start + t_win)
    ppg_ts = full["ppg_ts"][m]
    ppg_v = full["ppg_v"][m]

    # Filter PPG
    if len(ppg_v) > 5:
        nyq = FS_PPG / 2.0
        b, a = butter(2, [0.5 / nyq, 12.0 / nyq], btype="band")
        ppg_f = filtfilt(b, a, ppg_v)
    else:
        ppg_f = ppg_v.copy()

    # PPG peaks
    height_ppg = np.percentile(ppg_f, pct) if len(ppg_f) else 0
    ppg_idx, _ = find_peaks(ppg_f, distance=int(FS_PPG * 0.20), height=height_ppg)
    pt = ppg_ts[ppg_idx]

    # Physiological filter on peaks
    if len(pt) > 1:
        pp_raw = np.diff(pt)
        valid = (pp_raw >= MIN_INTERVAL) & (pp_raw <= MAX_INTERVAL)
        keep = np.ones(len(pt), dtype=bool)
        keep[1:] &= valid
        keep[:-1] &= valid
        pt = pt[keep]
        ppg_idx = ppg_idx[keep]

    pp = np.diff(pt) if len(pt) > 1 else np.array([])

    # --- Match beats ---
    TOL = 0.15
    ei, ppgi = [], []
    j = 0
    for i, r in enumerate(r_times):
        if j >= len(pt):
            break
        bj, bd = j, abs(pt[j] - r)
        for k in range(j, min(j + 5, len(pt))):
            d = abs(pt[k] - r)
            if d < bd:
                bd, bj = d, k
        if bd <= TOL:
            ei.append(i)
            ppgi.append(bj)
            j = bj + 1

    rr_m, pp_m = [], []
    for k in range(len(ei) - 1):
        if ei[k + 1] - ei[k] == 1 and ppgi[k + 1] - ppgi[k] == 1:
            rr_v = r_times[ei[k + 1]] - r_times[ei[k]]
            pp_v = pt[ppgi[k + 1]] - pt[ppgi[k]]
            if MIN_INTERVAL <= rr_v <= MAX_INTERVAL and MIN_INTERVAL <= pp_v <= MAX_INTERVAL:
                rr_m.append(rr_v)
                pp_m.append(pp_v)
    rr_m = np.array(rr_m) if rr_m else np.array([])
    pp_m = np.array(pp_m) if pp_m else np.array([])

    # --- Jitter ---
    ecg_dt = np.diff(ecg_t)
    ppg_dt = np.diff(ppg_ts)
    ecg_j = {"mean": ecg_dt.mean(), "sd": ecg_dt.std(), "min": ecg_dt.min(), "max": ecg_dt.max(),
             "n": len(ecg_dt), "fs": 1.0 / ecg_dt.mean() if len(ecg_dt) else 0}
    ppg_j = {"mean": ppg_dt.mean(), "sd": ppg_dt.std(), "min": ppg_dt.min(), "max": ppg_dt.max(),
             "n": len(ppg_dt), "fs": 1.0 / ppg_dt.mean() if len(ppg_dt) else 0}

    # --- Stats ---
    diff = rr_m - pp_m if len(rr_m) else np.array([])
    md = diff.mean() if len(diff) else 0
    sd = diff.std(ddof=1) if len(diff) > 1 else 0
    corr, pv = pearsonr(rr_m, pp_m) if len(rr_m) > 2 else (0, 1)
    sl, ic, rv, _, _ = linregress(rr_m, pp_m) if len(rr_m) > 2 else (0, 0, 0, 0, 0)

    n_rejected = max(0, len(ppg_idx) - len(pt)) if len(pt) else 0

    stats = {
        "n_ppg": len(pp),
        "n_rejected": n_rejected,
        "pp_mean": pp.mean() if len(pp) else 0,
        "pp_sd": pp.std() if len(pp) > 1 else 0,
        "n_match": len(rr_m),
        "md": md, "sd": sd,
        "rmse": np.sqrt((diff**2).mean()) if len(diff) else 0,
        "corr": corr, "pval": pv, "r2": rv**2,
        "slope": sl, "intercept": ic,
        "loa_upper": md + 1.96 * sd,
        "loa_lower": md - 1.96 * sd,
    }

    data = {
        "ecg_t": ecg_t, "ecg_mv": ecg_mv,
        "r_times": r_times, "r_peaks": r_peaks,
        "ppg_ts": ppg_ts, "ppg_v": ppg_v, "ppg_f": ppg_f,
        "pt": pt, "ppi": ppg_idx,
        "rr_m": rr_m, "pp_m": pp_m,
        "ecg_j": ecg_j, "ppg_j": ppg_j,
    }

    return data, stats


# ─── UI ───────────────────────────────────────────────────────────────
STYLESHEET = """
QMainWindow, QWidget { background-color: #0d1117; color: #c9d1d9; font-size: 13px; }
QPushButton { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
              border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { background: #30363d; }
QLabel#statVal { color: #4fc3f7; font-weight: bold; font-size: 14px; }
QLabel#statLabel { color: #8b949e; font-size: 11px; }
QLabel#sectionTitle { font-size: 18px; font-weight: bold; color: #4fc3f7; }
QSpinBox { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
           border-radius: 4px; padding: 3px; font-size: 13px; }
"""


class ECGPPGAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG vs PPG — Peak & Interval Analysis")
        self.setMinimumSize(1500, 960)
        self.setStyleSheet(STYLESHEET)

        self.full = load_full_data()
        self._t_start = 10
        self._t_win = 60
        self._ppg_pct = 70

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        # ─── Left: plots ──────────────────────────────────────────
        left = QWidget()
        l_layout = QVBoxLayout(left)
        l_layout.setSpacing(6)

        self.ecg_plot = pg.PlotWidget(title="ECG — R-peaks")
        self.ecg_plot.setLabel("left", "ECG", units="mV")
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ecg_plot.setMinimumHeight(150)

        self.ppg_plot = pg.PlotWidget(title="PPG — filtered + peaks")
        self.ppg_plot.setLabel("left", "PPG")
        self.ppg_plot.setLabel("bottom", "Time", units="s")
        self.ppg_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ppg_plot.setMinimumHeight(150)
        self.ppg_plot.setXLink(self.ecg_plot)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.corr_plot = pg.PlotWidget(title="Scatter — RR vs PP")
        self.corr_plot.setLabel("left", "PPG PP (ms)")
        self.corr_plot.setLabel("bottom", "ECG RR (ms)")
        self.corr_plot.showGrid(x=True, y=True, alpha=0.2)
        self.corr_plot.setMinimumHeight(170)
        self.ba_plot = pg.PlotWidget(title="Bland-Altman")
        self.ba_plot.setLabel("left", "Difference RR − PP (ms)")
        self.ba_plot.setLabel("bottom", "Mean of RR and PP (ms)")
        self.ba_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ba_plot.setMinimumHeight(170)
        grid.addWidget(self.corr_plot, 0, 0)
        grid.addWidget(self.ba_plot, 0, 1)

        # ─── Controls ─────────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(4)

        # Row 1: Start + Window
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        row1.addWidget(self._ctrl_label("Start:"))
        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setRange(0, int(MAX_START))
        self.start_slider.setValue(self._t_start)
        self.start_slider.setStyleSheet(self._slider_style())
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, int(MAX_START))
        self.start_spin.setValue(self._t_start)
        self.start_spin.setSuffix(" s")
        self.start_spin.setFixedWidth(72)
        row1.addWidget(self.start_slider, stretch=1)
        row1.addWidget(self.start_spin)

        row1.addSpacing(16)
        row1.addWidget(self._ctrl_label("Window:"))
        self.win_slider = QSlider(Qt.Horizontal)
        self.win_slider.setRange(int(MIN_WIN), int(MAX_WIN))
        self.win_slider.setValue(self._t_win)
        self.win_slider.setStyleSheet(self._slider_style())
        self.win_spin = QSpinBox()
        self.win_spin.setRange(int(MIN_WIN), int(MAX_WIN))
        self.win_spin.setValue(self._t_win)
        self.win_spin.setSuffix(" s")
        self.win_spin.setFixedWidth(72)
        row1.addWidget(self.win_slider, stretch=1)
        row1.addWidget(self.win_spin)

        ctrl.addLayout(row1)

        # Row 2: Threshold
        row2 = QHBoxLayout()
        row2.setSpacing(12)
        row2.addWidget(self._ctrl_label("PPG thresh:"))
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(1, 99)
        self.thresh_slider.setValue(self._ppg_pct)
        self.thresh_slider.setStyleSheet(self._slider_style())
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(1, 99)
        self.thresh_spin.setValue(self._ppg_pct)
        self.thresh_spin.setSuffix(" %")
        self.thresh_spin.setFixedWidth(62)
        self.n_peaks_label = QLabel("")
        self.n_peaks_label.setStyleSheet("color: #ff7043; font-size: 12px; font-weight: bold;")
        row2.addWidget(self.thresh_slider, stretch=1)
        row2.addWidget(self.thresh_spin)
        row2.addWidget(self.n_peaks_label)
        ctrl.addLayout(row2)

        self.start_slider.valueChanged.connect(self._on_start_changed)
        self.start_spin.valueChanged.connect(self._on_start_changed)
        self.win_slider.valueChanged.connect(self._on_win_changed)
        self.win_spin.valueChanged.connect(self._on_win_changed)
        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        self.thresh_spin.valueChanged.connect(self._on_thresh_changed)

        l_layout.addWidget(self.ecg_plot, stretch=1)
        l_layout.addWidget(self.ppg_plot, stretch=1)
        l_layout.addLayout(grid, stretch=1)
        l_layout.addLayout(ctrl)

        # ─── Right: stats ─────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(300)
        r_layout = QVBoxLayout(right)
        r_layout.setSpacing(5)

        title = QLabel("⚕ Comparison Results")
        title.setObjectName("sectionTitle")
        r_layout.addWidget(title)

        self.stat_labels = {}
        rows = [
            ("", None, ""), ("Window", None, "header"),
            ("Start", "win_start", "str"), ("Duration", "win_dur", "str"),
            ("", None, ""), ("ECG", None, "header"),
            ("R-peaks", "n_r", "int"), ("HR", "hr_ecg", "f1"),
            ("RR mean ± SD", "rr_str", "str"),
            ("", None, ""), ("PPG", None, "header"),
            ("Peaks", "n_ppg", "int"), ("Rejected", "n_rejected", "int"),
            ("HR", "hr_ppg", "f1"), ("PP mean ± SD", "pp_str", "str"),
            ("", None, ""), ("Comparison", None, "header"),
            ("Matched", "n_match", "int"), ("Mean diff (ms)", "md", "ms"),
            ("SD diff (ms)", "sd", "ms"), ("RMSE (ms)", "rmse", "ms"),
            ("Pearson r", "corr", "f4"), ("R²", "r2", "f4"),
            ("Upper LoA (ms)", "loa_upper", "ms"), ("Lower LoA (ms)", "loa_lower", "ms"),
            ("", None, ""), ("Sampling Jitter", None, "header"),
        ]
        self._build_rows(r_layout, rows)

        for label, key in [("ECG (150 Hz)", "ecg_j"), ("PPG (100 Hz)", "ppg_j")]:
            h = QLabel(label)
            h.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold; margin-top: 2px;")
            r_layout.addWidget(h)
            for txt, sk in [("Actual rate", "fs"), ("Mean dt (ms)", "mean"),
                            ("SD dt (ms)", "sd"), ("Min dt (ms)", "min"),
                            ("Max dt (ms)", "max"), ("Samples", "n")]:
                row = QHBoxLayout()
                l = QLabel(txt)
                l.setObjectName("statLabel")
                v = QLabel("—")
                v.setObjectName("statVal")
                v.setAlignment(Qt.AlignRight)
                row.addWidget(l); row.addWidget(v)
                r_layout.addLayout(row)
                self.stat_labels[f"j_{key}_{sk}"] = (v, key, sk)

        r_layout.addStretch()
        root.addWidget(left, stretch=1)
        root.addWidget(right)

        self._update()

    # ─── Helpers ──────────────────────────────────────────────────
    @staticmethod
    def _ctrl_label(text):
        l = QLabel(text)
        l.setStyleSheet("color: #8b949e; font-size: 12px;")
        return l

    @staticmethod
    def _slider_style():
        return (
            "QSlider::groove:horizontal { height: 6px; background: #21262d; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #4fc3f7; width: 16px; "
            "margin: -5px 0; border-radius: 8px; }"
            "QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 3px; }"
        )

    def _sync_spin_slider(self, spin, slider, value, from_spin):
        if from_spin:
            slider.blockSignals(True); slider.setValue(value); slider.blockSignals(False)
        else:
            spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)

    def _on_start_changed(self, value):
        from_spin = self.sender() is self.start_spin
        self._sync_spin_slider(self.start_spin, self.start_slider, value, from_spin)
        self._t_start = value
        self._update()

    def _on_win_changed(self, value):
        from_spin = self.sender() is self.win_spin
        self._sync_spin_slider(self.win_spin, self.win_slider, value, from_spin)
        self._t_win = value
        self._update()

    def _on_thresh_changed(self, value):
        from_spin = self.sender() is self.thresh_spin
        self._sync_spin_slider(self.thresh_spin, self.thresh_slider, value, from_spin)
        self._ppg_pct = value
        self._update()

    def _update(self):
        d, stats = window_and_process(self.full, float(self._t_start),
                                       float(self._t_win), self._ppg_pct)
        # Enrich stats
        rr = np.diff(d["r_times"])
        stats["_pct"] = self._ppg_pct
        stats["_t_start"] = self._t_start
        stats["_t_end"] = self._t_start + self._t_win
        stats["n_r"] = len(d["r_peaks"])
        stats["rr_mean"] = rr.mean() if len(rr) else 0
        stats["rr_sd"] = rr.std() if len(rr) > 1 else 0
        stats["hr_ecg"] = 60.0 / stats["rr_mean"] if stats["rr_mean"] > 0 else 0
        stats["_ecg_j"] = d["ecg_j"]
        stats["_ppg_j"] = d["ppg_j"]

        self._d = d
        self._plot_signals(d)
        self._plot_comparison(d, stats)
        self._update_stats(stats)
        self._print_stats(stats)

    def _build_rows(self, layout, rows):
        for label, key, fmt in rows:
            if label == "":
                layout.addSpacing(3)
                continue
            if fmt == "header":
                h = QLabel(label)
                h.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold; margin-top: 3px;")
                layout.addWidget(h)
                continue
            row = QHBoxLayout()
            l = QLabel(label)
            l.setObjectName("statLabel")
            v = QLabel("—")
            v.setObjectName("statVal")
            v.setAlignment(Qt.AlignRight)
            row.addWidget(l); row.addWidget(v)
            layout.addLayout(row)
            self.stat_labels[key] = (v, fmt)

    @staticmethod
    def _print_stats(stats):
        pp_mean = stats.get("pp_mean", 0)
        hr_ppg = 60.0 / pp_mean if pp_mean > 0 else 0
        print("\n============= ECG vs PPG ANALYSIS =============")
        print(f"  Window:              {stats.get('_t_start', '?')}s – {stats.get('_t_end', '?')}s")
        print(f"  Threshold:           {stats['_pct']}th percentile")
        print(f"  Physiological range: [{MIN_INTERVAL*1000:.0f}, {MAX_INTERVAL*1000:.0f}] ms  "
              f"({60/MAX_INTERVAL:.0f}–{60/MIN_INTERVAL:.0f} BPM)")
        print(f"  ECG R-peaks:         {stats['n_r']}  |  "
              f"HR = {stats['hr_ecg']:.1f} BPM  |  RR = {stats['rr_mean']*1000:.0f} ± {stats['rr_sd']*1000:.0f} ms")
        print(f"  PPG peaks:           {stats['n_ppg']}  |  rejected {stats.get('n_rejected',0)}  |  "
              f"HR = {hr_ppg:.1f} BPM  |  PP = {pp_mean*1000:.0f} ± {stats['pp_sd']*1000:.0f} ms")
        print(f"  Matched intervals:   {stats['n_match']}")
        print(f"  Mean diff (RR−PP):   {stats['md']*1000:.2f} ms")
        print(f"  SD of differences:   {stats['sd']*1000:.2f} ms")
        print(f"  RMSE:                {stats['rmse']*1000:.2f} ms")
        print(f"  Pearson r:           {stats['corr']:.4f}  (p={stats['pval']:.2e})")
        print(f"  R²:                  {stats['r2']:.4f}")
        print(f"  LoA:                 {stats['loa_lower']*1000:.1f} to {stats['loa_upper']*1000:.1f} ms")
        ej, pj = stats["_ecg_j"], stats["_ppg_j"]
        print(f"\n  Sampling Jitter:")
        print(f"    ECG: {ej['fs']:.1f} Hz  |  dt = {ej['mean']*1000:.2f} ± {ej['sd']*1000:.3f} ms  "
              f"[{ej['min']*1000:.2f}, {ej['max']*1000:.2f}]")
        print(f"    PPG: {pj['fs']:.1f} Hz  |  dt = {pj['mean']*1000:.2f} ± {pj['sd']*1000:.3f} ms  "
              f"[{pj['min']*1000:.2f}, {pj['max']*1000:.2f}]")
        print("================================================\n")

    def _plot_signals(self, d):
        for ax in [self.ecg_plot, self.ppg_plot, self.corr_plot, self.ba_plot]:
            ax.clear()

        self.ecg_plot.plot(d["ecg_t"], d["ecg_mv"], pen=pg.mkPen("#4fc3f7", width=1))
        if len(d["r_peaks"]):
            self.ecg_plot.addItem(pg.ScatterPlotItem(
                d["r_times"], d["ecg_mv"][d["r_peaks"]],
                pen=pg.mkPen(None), brush=pg.mkBrush(255, 60, 60), size=10, symbol="o"))

        if len(d["ppg_f"]):
            self.ppg_plot.plot(d["ppg_ts"], d["ppg_f"], pen=pg.mkPen("#ff7043", width=1.2))
        if len(d["pt"]):
            self.ppg_plot.addItem(pg.ScatterPlotItem(
                d["pt"], d["ppg_f"][d["ppi"]],
                pen=pg.mkPen(None), brush=pg.mkBrush(255, 60, 60), size=10, symbol="o"))

        self.ecg_plot.autoRange()
        self.n_peaks_label.setText(f"  {len(d['ppi'])} peaks")

    def _plot_comparison(self, d, stats):
        rr, pp = d["rr_m"], d["pp_m"]
        if len(rr) < 3:
            return
        rr_ms, pp_ms = rr * 1000, pp * 1000
        self.corr_plot.plot(rr_ms, pp_ms, pen=None, symbol="o", symbolSize=6,
                            symbolBrush="#4fc3f7", symbolPen="#4fc3f7")
        lo = min(rr_ms.min(), pp_ms.min()) - 15
        hi = max(rr_ms.max(), pp_ms.max()) + 15
        self.corr_plot.plot([lo, hi], [lo, hi], pen=pg.mkPen("white", width=1, style=Qt.DashLine))
        xf = np.array([lo, hi])
        yf = (stats["slope"] * xf / 1000 + stats["intercept"]) * 1000
        self.corr_plot.plot(xf, yf, pen=pg.mkPen("#ff7043", width=1.5))
        self.corr_plot.setTitle(
            f"Scatter  (r = {stats['corr']:.3f},  y = {stats['slope']:.2f}x + {stats['intercept']*1000:.0f})")
        self.corr_plot.setXRange(lo, hi); self.corr_plot.setYRange(lo, hi)

        mean_v = (rr_ms + pp_ms) / 2
        diff_v = (rr - pp) * 1000
        self.ba_plot.plot(mean_v, diff_v, pen=None, symbol="o", symbolSize=6,
                          symbolBrush="#4fc3f7", symbolPen="#4fc3f7")
        md = stats["md"] * 1000; sd = stats["sd"] * 1000
        self.ba_plot.addItem(pg.InfiniteLine(pos=md, angle=0, pen=pg.mkPen("red", width=1.5)))
        self.ba_plot.addItem(pg.InfiniteLine(pos=md + 1.96 * sd, angle=0,
                                              pen=pg.mkPen("gray", width=1, style=Qt.DashLine)))
        self.ba_plot.addItem(pg.InfiniteLine(pos=md - 1.96 * sd, angle=0,
                                              pen=pg.mkPen("gray", width=1, style=Qt.DashLine)))
        self.ba_plot.setTitle(
            f"Bland-Altman  (bias = {md:.1f}, LoA = [{md-1.96*sd:.1f}, {md+1.96*sd:.1f}])")

    def _update_stats(self, stats):
        vals = {
            "win_start": f"{self._t_start}s", "win_dur": f"{self._t_win}s",
            "hr_ppg": 60.0 / stats["pp_mean"] if stats["pp_mean"] > 0 else 0,
            "rr_str": f"{stats['rr_mean']*1000:.0f} ± {stats['rr_sd']*1000:.0f} ms" if stats["n_r"] else "—",
            "pp_str": f"{stats['pp_mean']*1000:.0f} ± {stats['pp_sd']*1000:.0f} ms" if stats["n_ppg"] else "—",
        }
        vals.update(stats)

        for key, item in self.stat_labels.items():
            if key.startswith("j_"):
                suffix = key[2:]  # e.g. "ecg_j_fs" or "ppg_j_mean"
                widget, _, _ = item
                # Known subkeys
                for sk in ("fs", "mean", "sd", "min", "max", "n"):
                    if suffix.endswith(sk):
                        jkey = suffix[:-(len(sk) + 1)]  # remove "_sk"
                        break
                else:
                    continue
                j = vals["_ecg_j"] if jkey == "ecg_j" else vals["_ppg_j"]
                v = j[sk]
                if sk == "fs":
                    widget.setText(f"{v:.1f} Hz")
                elif sk in ("mean", "sd", "min", "max"):
                    widget.setText(f"{v*1000:.3f} ms")
                else:
                    widget.setText(f"{v}")
                continue

            widget, fmt = item
            v = vals.get(key, 0)
            if fmt == "int":     widget.setText(f"{v:.0f}")
            elif fmt == "f1":    widget.setText(f"{v:.1f}")
            elif fmt == "f4":    widget.setText(f"{v:.4f}")
            elif fmt == "ms":    widget.setText(f"{v*1000:.2f}")
            elif fmt == "str":   widget.setText(str(v))
            else:                widget.setText(f"{v:.4f}" if 0 < v < 1 else f"{v:.2f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ECGPPGAnalyzer()
    w.show()
    sys.exit(app.exec_())
