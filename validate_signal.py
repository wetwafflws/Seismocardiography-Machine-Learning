#!/usr/bin/env python3
"""
Signal Sampling Rate, Jitter, and Beat Data Validation Script
=============================================================
This script performs a rigorous verification of hardware sampling rates, timing jitter,
host-device synchronization, and cardiac inter-beat intervals (IBI) for seismocardiography (SCG)
and photoplethysmography (PPG) multisensor records.

It handles mixed-row CSV formats containing:
  - 3-axis accelerometer (SCG) data (expected at 256 Hz)
  - PPG raw data (expected at 100 Hz)
  - Beat event markers (expected asynchronously upon detection)

Author: Antigravity
Date: June 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ANSI Colors for beautiful terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}\n{title.center(80)}\n{'='*80}{Colors.ENDC}")

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {title} ---{Colors.ENDC}")

def print_metric(label, value, unit="", level="info"):
    color = Colors.ENDC
    if level == "success":
        color = Colors.GREEN
    elif level == "warning":
        color = Colors.WARNING
    elif level == "error":
        color = Colors.FAIL
    elif level == "highlight":
        color = Colors.BLUE
        
    print(f"  {label:<50} : {color}{Colors.BOLD}{value}{Colors.ENDC} {unit}")

def analyze_stream(name, timestamps, host_timestamps, expected_fs, is_scg=False):
    """
    Analyzes sampling rate, jitter, packet loss, and quantization effects for a stream.
    """
    if len(timestamps) < 2:
        return None

    # Calculate intervals
    intervals = np.diff(timestamps)
    host_intervals = np.diff(host_timestamps)
    
    # Basic counts
    n_samples = len(timestamps)
    expected_interval = 1000.0 / expected_fs # in ms
    
    # 1. Device Timer Metrics
    mean_interval = np.mean(intervals)
    actual_fs = 1000.0 / mean_interval
    fs_error_pct = abs(actual_fs - expected_fs) / expected_fs * 100.0
    
    std_jitter = np.std(intervals)
    # Root Mean Square of Successive Differences (RMSSD) of sampling intervals
    rmssd_jitter = np.sqrt(np.mean(np.diff(intervals) ** 2))
    
    min_int, max_int = np.min(intervals), np.max(intervals)
    median_int = np.median(intervals)
    
    # 2. Host Timer Metrics
    host_mean_interval = np.mean(host_intervals)
    host_actual_fs = 1000.0 / host_mean_interval
    host_std_jitter = np.std(host_intervals)
    host_rmssd_jitter = np.sqrt(np.mean(np.diff(host_intervals) ** 2))
    host_min_int, host_max_int = np.min(host_intervals), np.max(host_intervals)
    
    # 3. Quantization and Perfect Sampling Checks
    counter = Counter(intervals)
    
    # Missing samples & duplicates
    duplicates = np.sum(intervals == 0)
    duplicates_pct = duplicates / len(intervals) * 100.0
    
    if is_scg:
        # SCG at 256 Hz has a theoretical period of 3.90625 ms.
        # With integer ms timestamps, a perfectly regular clock will alternate
        # between 3 ms (9.375% probability) and 4 ms (90.625% probability).
        # Theoretical quantization jitter (standard deviation) is approx 0.2915 ms.
        perfect_count = counter[3] + counter[4]
        pct_perfect = perfect_count / len(intervals) * 100.0
        
        # Missing samples: intervals >= 6 ms (since 3.90625 * 1.5 = 5.86 ms)
        missing_mask = intervals >= 6
        missing_count = 0
        if np.any(missing_mask):
            missing_count = int(np.sum(np.round(intervals[missing_mask] / expected_interval) - 1))
            
        theory_notes = "256 Hz rounding to integer ms results in theoretical alternating 3/4 ms intervals."
    else:
        # PPG at 100 Hz has an exact 10 ms interval.
        # Perfectly regular clock should yield exactly 10 ms.
        perfect_count = counter[10]
        pct_perfect = perfect_count / len(intervals) * 100.0
        
        # Missing samples: intervals >= 15 ms (since 10 * 1.5 = 15 ms)
        missing_mask = intervals >= 15
        missing_count = 0
        if np.any(missing_mask):
            missing_count = int(np.sum(np.round(intervals[missing_mask] / expected_interval) - 1))
            
        theory_notes = "100 Hz rounds perfectly to 10 ms integer intervals."

    missing_pct = missing_count / (n_samples + missing_count) * 100.0 if (n_samples + missing_count) > 0 else 0.0

    return {
        'name': name,
        'n_samples': n_samples,
        'expected_fs': expected_fs,
        'expected_interval': expected_interval,
        'intervals': intervals,
        'host_intervals': host_intervals,
        'mean_interval': mean_interval,
        'actual_fs': actual_fs,
        'fs_error_pct': fs_error_pct,
        'std_jitter': std_jitter,
        'rmssd_jitter': rmssd_jitter,
        'min_int': min_int,
        'max_int': max_int,
        'median_int': median_int,
        'counter': counter,
        'pct_perfect': pct_perfect,
        'duplicates': duplicates,
        'duplicates_pct': duplicates_pct,
        'missing_count': missing_count,
        'missing_pct': missing_pct,
        'theory_notes': theory_notes,
        # Host metrics
        'host_mean_interval': host_mean_interval,
        'host_actual_fs': host_actual_fs,
        'host_std_jitter': host_std_jitter,
        'host_rmssd_jitter': host_rmssd_jitter,
        'host_min_int': host_min_int,
        'host_max_int': host_max_int,
    }

def analyze_beats(beat_df):
    """
    Validates beat interval data, comparing device-calculated IBI to calculated differences.
    """
    if len(beat_df) < 2:
        return None

    beat_ts = beat_df['timestamp_ms'].to_numpy()
    beat_host = beat_df['host_time_ms'].to_numpy()
    reported_ibi = beat_df['ibi_ms'].to_numpy()

    # Calculate actual intervals
    calc_ibi_device = np.diff(beat_ts)
    calc_ibi_host = np.diff(beat_host)
    
    # Align reported_ibi with calculated
    # Row k has ibi_ms which represents the interval from the PREVIOUS beat.
    # Therefore, reported_ibi[1:] aligns with calc_ibi_device[0:]
    reported_ibi_aligned = reported_ibi[1:]
    
    # Create mask for valid (non-NaN) reported IBI values
    valid_mask = ~np.isnan(reported_ibi_aligned)
    
    # Accuracy check
    if np.any(valid_mask):
        errors_device = calc_ibi_device[valid_mask] - reported_ibi_aligned[valid_mask]
        mae_device = np.mean(np.abs(errors_device))
        max_err_device = np.max(np.abs(errors_device))
        
        errors_host = calc_ibi_host[valid_mask] - reported_ibi_aligned[valid_mask]
        mae_host = np.mean(np.abs(errors_host))
        max_err_host = np.max(np.abs(errors_host))
    else:
        mae_device, max_err_device = np.nan, np.nan
        mae_host, max_err_host = np.nan, np.nan

    # HRV Metrics (from Device Timestamps which are reliable)
    # Mean Heart Rate
    mean_ibi = np.mean(calc_ibi_device)
    mean_hr = 60000.0 / mean_ibi
    
    # SDNN: Standard Deviation of NN (Normal-to-Normal) intervals
    sdnn = np.std(calc_ibi_device)
    
    # RMSSD: Root Mean Square of Successive Differences of NN intervals
    rmssd = np.sqrt(np.mean(np.diff(calc_ibi_device) ** 2))
    
    min_hr = 60000.0 / np.max(calc_ibi_device)
    max_hr = 60000.0 / np.min(calc_ibi_device)

    return {
        'n_beats': len(beat_df),
        'beat_ts': beat_ts,
        'calc_ibi_device': calc_ibi_device,
        'calc_ibi_host': calc_ibi_host,
        'reported_ibi': reported_ibi_aligned,
        'mean_hr': mean_hr,
        'min_hr': min_hr,
        'max_hr': max_hr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'mae_device': mae_device,
        'max_err_device': max_err_device,
        'mae_host': mae_host,
        'max_err_host': max_err_host,
    }

def print_stream_report(r):
    if r is None:
        print(f"  No data or insufficient samples to analyze.")
        return

    expected_str = f"{r['expected_fs']} Hz ({r['expected_interval']:.3f} ms)"
    actual_device_str = f"{r['actual_fs']:.3f} Hz ({r['mean_interval']:.4f} ms)"
    actual_host_str = f"{r['host_actual_fs']:.3f} Hz ({r['host_mean_interval']:.4f} ms)"
    
    print_metric("Expected Sampling Rate", expected_str)
    print_metric("Actual Sampling Rate (Device Timer)", actual_device_str, level="highlight")
    print_metric("Actual Sampling Rate (Host Clock)", actual_host_str)
    
    error_level = "success" if r['fs_error_pct'] < 0.1 else ("warning" if r['fs_error_pct'] < 1.0 else "error")
    print_metric("Sampling Frequency Error (Device)", f"{r['fs_error_pct']:.4f}", "%", level=error_level)
    
    print_metric("Jitter SD (Device Timer)", f"{r['std_jitter']:.4f}", "ms", level="success" if r['std_jitter'] < 0.5 else "warning")
    print_metric("Jitter SD (Host Clock)", f"{r['host_std_jitter']:.4f}", "ms", level="warning" if r['host_std_jitter'] > 1.0 else "info")
    
    print_metric("RMSSD Jitter (Device Timer)", f"{r['rmssd_jitter']:.4f}", "ms")
    print_metric("RMSSD Jitter (Host Clock)", f"{r['host_rmssd_jitter']:.4f}", "ms")
    
    print_metric("Device Interval Range (Min / Max)", f"{r['min_int']} / {r['max_int']}", "ms")
    print_metric("Host Interval Range (Min / Max)", f"{r['host_min_int']} / {r['host_max_int']}", "ms")
    
    # Quantization analysis
    quant_level = "success" if r['pct_perfect'] > 98.0 else ("warning" if r['pct_perfect'] > 90.0 else "error")
    perfect_label = "Perfect 3ms or 4ms Intervals" if r['expected_fs'] == 256 else "Perfect 10ms Intervals"
    print_metric(perfect_label, f"{r['pct_perfect']:.2f}", "%", level=quant_level)
    print(f"    [Note] {r['theory_notes']}")
    
    # Packet loss
    loss_level = "success" if r['missing_count'] == 0 else ("warning" if r['missing_pct'] < 0.5 else "error")
    print_metric("Estimated Missed Samples (Packet Loss)", f"{r['missing_count']} ({r['missing_pct']:.3f}%)", level=loss_level)
    
    # Duplicates
    dup_level = "success" if r['duplicates'] == 0 else "warning"
    print_metric("Duplicate Timestamps (dt = 0)", f"{r['duplicates']} ({r['duplicates_pct']:.3f}%)", level=dup_level)

def main():
    parser = argparse.ArgumentParser(description="Validate sampling rate and jitter of PPG and SCG signals.")
    parser.add_argument('csv_path', type=str, nargs='?', 
                        default='SUBJECT_Data/2026-05-18/MRAH_201640.csv',
                        help="Path to the signal CSV file.")
    parser.add_argument('--no-plot', action='store_true', help="Disable plot rendering.")
    args = parser.parse_args()

    csv_path = args.csv_path

    if not os.path.exists(csv_path):
        print(f"{Colors.FAIL}Error: File not found at {csv_path}{Colors.ENDC}")
        sys.exit(1)

    print_header(f"Seismocardiography & PPG Signal Validator\nFile: {os.path.basename(csv_path)}")

    # 1. Parse Metadata from CSV Headers
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                parts = line[1:].strip().split(',', 1)
                if len(parts) == 2:
                    key, val = parts[0].strip(), parts[1].strip()
                    metadata[key] = val
            else:
                break # First line without '#' starts the CSV headers
                
    print_section("Recording Metadata")
    for k, v in metadata.items():
        print(f"  {k:<30} : {Colors.BOLD}{v}{Colors.ENDC}")

    # Expected frequencies from metadata (or defaults)
    expected_scg_fs = int(metadata.get('sample_rate_scg_hz', 256))
    expected_ppg_fs = int(metadata.get('sample_rate_ppg_hz', 100))

    # 2. Load CSV Data
    print(f"\n{Colors.BLUE}Loading dataset...{Colors.ENDC}")
    df = pd.read_csv(csv_path, comment='#')
    print(f"Loaded {Colors.BOLD}{len(df):,}{Colors.ENDC} rows.")

    # 3. Extract Streams
    print(f"{Colors.BLUE}Extracting individual data streams...{Colors.ENDC}")
    
    # SCG: Accelerometer records have non-null 'x_g' (and y_g, z_g)
    scg_df = df[df['x_g'].notna()]
    scg_ts = scg_df['timestamp_ms'].to_numpy()
    scg_host = scg_df['host_time_ms'].to_numpy()
    
    # PPG: Photoplethysmography records have non-null 'ppg_raw'
    ppg_df = df[df['ppg_raw'].notna()]
    ppg_ts = ppg_df['timestamp_ms'].to_numpy()
    ppg_host = ppg_df['host_time_ms'].to_numpy()
    
    # Beats: Beat events have 'beat_event' == 1
    beat_df = df[df['beat_event'] == 1]

    # Print streams overview
    print(f"  - SCG (3-Axis Accelerometer) stream: {Colors.BOLD}{len(scg_ts):,}{Colors.ENDC} samples")
    print(f"  - PPG (Photoplethysmography) stream: {Colors.BOLD}{len(ppg_ts):,}{Colors.ENDC} samples")
    print(f"  - Heart Beat (Asynchronous) events : {Colors.BOLD}{len(beat_df)}{Colors.ENDC} beats")

    # 4. Perform Detailed Stream Analysis
    scg_report = analyze_stream("SCG (Accelerometer)", scg_ts, scg_host, expected_scg_fs, is_scg=True)
    ppg_report = analyze_stream("PPG (Photoplethysmography)", ppg_ts, ppg_host, expected_ppg_fs, is_scg=False)
    beat_report = analyze_beats(beat_df)

    # 5. Display Reports
    print_header("TIMING & JITTER VALIDATION REPORT")

    print_section(f"SCG Stream (Expected {expected_scg_fs} Hz)")
    print_stream_report(scg_report)

    print_section(f"PPG Stream (Expected {expected_ppg_fs} Hz)")
    print_stream_report(ppg_report)

    print_section("Beat Event & Inter-Beat Interval (IBI) Verification")
    if beat_report:
        print_metric("Total Detected Beat Events", beat_report['n_beats'])
        print_metric("Mean Heart Rate (Device)", f"{beat_report['mean_hr']:.2f}", "BPM", level="highlight")
        print_metric("Heart Rate Range (Min / Max)", f"{beat_report['min_hr']:.1f} / {beat_report['max_hr']:.1f}", "BPM")
        
        # HRV metrics
        print_metric("Heart Rate Variability SDNN (Device)", f"{beat_report['sdnn']:.2f}", "ms")
        print_metric("Heart Rate Variability RMSSD (Device)", f"{beat_report['rmssd']:.2f}", "ms")
        
        # Device IBI validation
        dev_err_level = "success" if beat_report['mae_device'] < 0.01 else "warning"
        print_metric("Device IBI vs Timestamp Diff MAE", f"{beat_report['mae_device']:.4f}", "ms", level=dev_err_level)
        print_metric("Device IBI vs Timestamp Diff Max Error", f"{beat_report['max_err_device']:.4f}", "ms", level=dev_err_level)
        print("    [Info] 0.00 ms error confirms perfect hardware timing of microcontroller peak-detection.")
        
        # Host IBI validation
        host_err_level = "warning" if beat_report['mae_host'] > 1.0 else "info"
        print_metric("Host IBI vs Timestamp Diff MAE (USB Jitter)", f"{beat_report['mae_host']:.4f}", "ms", level=host_err_level)
        print_metric("Host IBI vs Timestamp Diff Max Error (USB Jitter)", f"{beat_report['max_err_host']:.4f}", "ms", level=host_err_level)
        print("    [Info] The difference shows transmission/buffer delay variance introduced by the OS/USB connection.")
    else:
        print("  No beat event data found in this record.")

    # 6. Device-Host Synchronization & Clock Drift Analysis
    print_section("Device-Host Synchronization & Clock Drift")
    if len(scg_ts) > 2:
        # Duration on device timer
        device_duration_sec = (scg_ts[-1] - scg_ts[0]) / 1000.0
        # Duration on host timer
        host_duration_sec = (scg_host[-1] - scg_host[0]) / 1000.0
        
        drift_ms = (scg_host[-1] - scg_host[0]) - (scg_ts[-1] - scg_ts[0])
        drift_rate_ms_hr = drift_ms / (device_duration_sec / 3600.0)
        drift_ppm = drift_ms / (device_duration_sec * 1000.0) * 1e6
        
        print_metric("Device Total Duration", f"{device_duration_sec:.2f} s ({device_duration_sec/60.0:.2f} min)")
        print_metric("Host Total Duration", f"{host_duration_sec:.2f} s ({host_duration_sec/60.0:.2f} min)")
        
        drift_level = "success" if abs(drift_ppm) < 50 else ("warning" if abs(drift_ppm) < 500 else "error")
        print_metric("Net Synchronization Drift", f"{drift_ms:.2f}", "ms", level=drift_level)
        print_metric("Clock Drift Rate", f"{drift_rate_ms_hr:.2f} ms/hour ({drift_ppm:.2f} ppm)", level=drift_level)
        print("    [Info] Clock drift values under 100 ppm are normal for standard quartz oscillators.")

    # 7. Generate Visualizations using Matplotlib
    if not args.no_plot:
        print(f"\n{Colors.BLUE}Generating visual diagnostic charts...{Colors.ENDC}")
        
        # Enable professional dark grid styling
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Timing & Jitter Diagnostic Dashboard\nRecord: {os.path.basename(csv_path)}", 
                     fontsize=16, fontweight='bold', color='#2c3e50')

        # Colors for plots
        dev_color = '#1f77b4'  # Professional blue
        host_color = '#e74c3c' # Professional soft red
        accent_color = '#2ecc71' # Green

        # --- SUBPLOT 1: SCG Timing Interval Histogram ---
        ax = axs[0, 0]
        if scg_report:
            # We want to show both device and host intervals
            # For device, it's mostly 3 and 4 ms due to quantization
            # We'll use custom bins for device since it is discrete
            bins_dev = np.arange(0, 10, 0.5) - 0.25
            ax.hist(scg_report['intervals'], bins=bins_dev, color=dev_color, alpha=0.7, 
                    label=f"Device Timer (Jitter SD: {scg_report['std_jitter']:.3f} ms)", rwidth=0.8)
            ax.hist(scg_report['host_intervals'], bins=np.arange(0, 10, 0.25), color=host_color, alpha=0.5, 
                    label=f"Host Timer (Jitter SD: {scg_report['host_std_jitter']:.3f} ms)")
            
            ax.set_title("SCG (Accelerometer) Sampling Intervals (Expected ~3.9 ms)", fontweight='bold')
            ax.set_xlabel("Interval Duration (ms)")
            ax.set_ylabel("Occurrences")
            ax.set_xlim(0, 8)
            ax.legend(frameon=True, facecolor='white', framealpha=0.9)
            
            # Add summary text box
            textstr = '\n'.join((
                f"SCG Expected fs: {expected_scg_fs} Hz",
                f"Device mean fs: {scg_report['actual_fs']:.2f} Hz",
                f"Perfect alternating 3/4 ms: {scg_report['pct_perfect']:.1f}%",
                f"Estimated missed samples: {scg_report['missing_count']}"
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

        # --- SUBPLOT 2: PPG Timing Interval Histogram ---
        ax = axs[0, 1]
        if ppg_report:
            # PPG is expected at 10 ms.
            bins_ppg = np.arange(5, 15, 0.5) - 0.25
            ax.hist(ppg_report['intervals'], bins=bins_ppg, color=dev_color, alpha=0.7, 
                    label=f"Device Timer (Jitter SD: {ppg_report['std_jitter']:.3f} ms)", rwidth=0.8)
            ax.hist(ppg_report['host_intervals'], bins=np.arange(5, 15, 0.25), color=host_color, alpha=0.5, 
                    label=f"Host Timer (Jitter SD: {ppg_report['host_std_jitter']:.3f} ms)")
            
            ax.set_title("PPG Sampling Intervals (Expected 10.0 ms)", fontweight='bold')
            ax.set_xlabel("Interval Duration (ms)")
            ax.set_ylabel("Occurrences")
            ax.set_xlim(5, 15)
            ax.legend(frameon=True, facecolor='white', framealpha=0.9)
            
            # Add summary text box
            textstr = '\n'.join((
                f"PPG Expected fs: {expected_ppg_fs} Hz",
                f"Device mean fs: {ppg_report['actual_fs']:.2f} Hz",
                f"Perfect 10 ms intervals: {ppg_report['pct_perfect']:.1f}%",
                f"Estimated missed samples: {ppg_report['missing_count']}"
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

        # --- SUBPLOT 3: Host vs Device Clock Drift ---
        ax = axs[1, 0]
        if len(scg_ts) > 2:
            # Downsample for visualization speed
            step = max(1, len(scg_ts) // 2000)
            scg_time_min = (scg_ts[::step] - scg_ts[0]) / 60000.0
            
            # Calculate drift at each point
            scg_drift_ms = (scg_host[::step] - scg_host[0]) - (scg_ts[::step] - scg_ts[0])
            
            ax.plot(scg_time_min, scg_drift_ms, color='#8e44ad', linewidth=2, label="Accumulated Drift")
            
            # Fit a linear line to show trend
            slope, intercept = np.polyfit(scg_time_min, scg_drift_ms, 1)
            ax.plot(scg_time_min, slope * scg_time_min + intercept, color='#2c3e50', linestyle='--', 
                    label=f"Drift Rate: {slope * 60.0:.2f} ms/hour")
            
            ax.set_title("Accumulated Clock Drift (Host Time - Device Time)", fontweight='bold')
            ax.set_xlabel("Elapsed Time (minutes)")
            ax.set_ylabel("Clock Sync Drift (ms)")
            ax.legend(frameon=True, facecolor='white', framealpha=0.9)

        # --- SUBPLOT 4: Inter-Beat Interval (IBI) Tracking & Host Jitter ---
        ax = axs[1, 1]
        if beat_report:
            beat_time_min = (beat_report['beat_ts'][1:] - beat_report['beat_ts'][0]) / 60000.0
            
            ax.plot(beat_time_min, beat_report['reported_ibi'], color=accent_color, linewidth=2, 
                    label="Clean Device IBI / T-diff")
            
            # Scatter plot of host intervals to show visual jitter
            ax.scatter(beat_time_min, beat_report['calc_ibi_host'], color=host_color, s=15, alpha=0.6,
                       label=f"Host-observed IBI (MAE: {beat_report['mae_host']:.1f} ms)")
            
            ax.set_title("Heart IBI & Host-observed Transmission Jitter", fontweight='bold')
            ax.set_xlabel("Elapsed Time (minutes)")
            ax.set_ylabel("Inter-Beat Interval (ms)")
            ax.legend(frameon=True, facecolor='white', framealpha=0.9)
            
            # Y-axis zoom to show details comfortably
            q25, q75 = np.percentile(beat_report['reported_ibi'], [25, 75])
            iqr = q75 - q25
            ax.set_ylim(max(300, q25 - 2*iqr), min(1800, q75 + 2*iqr))

        plt.tight_layout()
        
        # Save as a PNG file artifact
        output_png = 'signal_validation_results.png'
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        print(f"\n{Colors.GREEN}{Colors.BOLD}Success! Beautiful validation charts saved to: {output_png}{Colors.ENDC}")
        
        # Show interactive GUI window if backend is available
        try:
            plt.show()
        except Exception:
            pass # Non-interactive environments will fail silently here, which is fine since PNG is saved

    print_header("VALIDATION COMPLETED")

if __name__ == "__main__":
    main()
