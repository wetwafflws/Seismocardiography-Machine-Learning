from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="SCG CSV Viewer", layout="wide")
st.title("SCG CSV Viewer")
st.caption("Load a saved SCG CSV and inspect waveform, beats, timing, and sampling rate.")


REQUIRED_COLS = ["timestamp_ms", "x_g", "y_g", "z_g", "beat_event"]


def _read_csv_from_source(uploaded_file, selected_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(selected_path)


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


with st.sidebar:
    st.header("Data Source")
    source_mode = st.radio("Choose source", ["Workspace CSV", "Upload CSV"], index=0)

    uploaded = None
    selected_file_path = ""

    if source_mode == "Workspace CSV":
        search_dir = st.text_input("Search folder", value=".")
        path_obj = Path(search_dir)
        if path_obj.exists() and path_obj.is_dir():
            candidates = sorted(path_obj.glob("*.csv"))
            if candidates:
                selected = st.selectbox("CSV file", [str(p) for p in candidates])
                selected_file_path = selected
            else:
                st.warning("No CSV files found in that folder.")
        else:
            st.warning("Folder does not exist.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    expected_hz = st.number_input("Expected sample rate (Hz)", min_value=1.0, value=256.0, step=1.0)


if uploaded is None and not selected_file_path:
    st.info("Select or upload a CSV file to begin.")
    st.stop()

try:
    raw_df = _read_csv_from_source(uploaded, selected_file_path)
    df = _prepare_df(raw_df)
except Exception as exc:
    st.error(f"Failed to load CSV: {exc}")
    st.stop()


scg_df = df[df[["x_g", "y_g", "z_g"]].notna().any(axis=1)].copy()
beats_df = df[df["beat_event"] == 1].copy()

if scg_df.empty:
    st.error("No SCG rows found (x_g/y_g/z_g are empty for all rows).")
    st.stop()


duration_s, actual_hz = _compute_rate(scg_df)
diff_hz = actual_hz - expected_hz
pct = (diff_hz / expected_hz * 100.0) if expected_hz > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("SCG Samples", f"{len(scg_df):,}")
m2.metric("Beats", f"{len(beats_df):,}")
m3.metric("Actual Rate", f"{actual_hz:.2f} Hz")
m4.metric("Rate Error", f"{diff_hz:+.2f} Hz", f"{pct:+.2f}%")

st.caption(f"Capture duration: {duration_s:.2f} s")


t0_ms = float(scg_df["timestamp_ms"].iloc[0])
scg_df["time_s"] = (scg_df["timestamp_ms"] - t0_ms) / 1000.0
beats_df["time_s"] = (beats_df["timestamp_ms"] - t0_ms) / 1000.0 if not beats_df.empty else np.array([])

max_t = float(scg_df["time_s"].iloc[-1])

with st.sidebar:
    st.header("View Options")
    channels = st.multiselect("Channels", ["x_g", "y_g", "z_g"], default=["x_g", "y_g", "z_g"])
    show_beats = st.checkbox("Show beat markers", value=True)
    y_locked = st.checkbox("Lock y-axis to +/-3 g", value=True)

    if max_t > 0:
        t_window = st.slider("Time range (s)", min_value=0.0, max_value=max_t, value=(0.0, max_t), step=0.1)
    else:
        t_window = (0.0, 0.0)

if not channels:
    st.warning("Select at least one channel.")
    st.stop()


start_s, end_s = t_window
view_scg = scg_df[(scg_df["time_s"] >= start_s) & (scg_df["time_s"] <= end_s)]
view_beats = beats_df[(beats_df["time_s"] >= start_s) & (beats_df["time_s"] <= end_s)] if not beats_df.empty else beats_df

fig = go.Figure()
color_map = {"x_g": "#00e5ff", "y_g": "#a78bfa", "z_g": "#2ed573"}

for ch in channels:
    fig.add_trace(
        go.Scatter(
            x=view_scg["time_s"],
            y=view_scg[ch],
            mode="lines",
            name=ch,
            line=dict(width=1.5, color=color_map.get(ch, None)),
        )
    )

if show_beats and not view_beats.empty:
    for bt in view_beats["time_s"].to_numpy():
        fig.add_vline(x=float(bt), line_width=1, line_dash="dash", line_color="#ff4757", opacity=0.6)

fig.update_layout(
    title="SCG Waveform",
    xaxis_title="Time (s)",
    yaxis_title="Acceleration (g)",
    template="plotly_dark",
    height=520,
    margin=dict(l=20, r=20, t=50, b=20),
)

if y_locked:
    fig.update_yaxes(range=[-3.0, 3.0])

st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3 = st.tabs(["Beat Intervals", "Data Table", "Channel Stats"])

with tab1:
    if len(beats_df) >= 2:
        beat_times = beats_df["timestamp_ms"].to_numpy(dtype=float)
        rr_s = np.diff(beat_times) / 1000.0
        bpm = np.divide(60.0, rr_s, out=np.zeros_like(rr_s), where=rr_s > 0)

        interval_df = pd.DataFrame(
            {
                "beat_index": np.arange(1, len(rr_s) + 1),
                "rr_interval_s": rr_s,
                "instant_bpm": bpm,
            }
        )

        st.dataframe(interval_df, use_container_width=True, height=260)

        rr_fig = go.Figure()
        rr_fig.add_trace(
            go.Scatter(
                x=interval_df["beat_index"],
                y=interval_df["rr_interval_s"],
                mode="lines+markers",
                name="RR interval (s)",
                line=dict(color="#ff4757", width=1.5),
            )
        )
        rr_fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(rr_fig, use_container_width=True)
    else:
        st.info("Need at least 2 beat events for interval analysis.")

with tab2:
    st.dataframe(view_scg[["timestamp_ms", "time_s", "x_g", "y_g", "z_g"]], use_container_width=True, height=380)

with tab3:
    stats_df = view_scg[channels].describe().T
    st.dataframe(stats_df, use_container_width=True)
