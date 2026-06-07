#!/usr/bin/env python3
"""
Simplified Hardware Timing Batch Validator
==========================================
This script recursively processes all CSV files inside 'SUBJECT_Data/' to determine
whether the hardware is sampling data correctly. 

It generates a simplified summary of the sampling performance based exclusively on
the microcontroller device timestamps, omitting host timestamps and patient metadata.

Outputs:
  - Data_Validation/validation_summary.csv
  - Data_Validation/validation_summary.png (Timing & Jitter Diagnostics)

Author: Antigravity
Date: June 2026
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_metadata(csv_path):
    """Parses expected sample rates from the CSV comment headers."""
    expected_scg = 256
    expected_ppg = 100
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                parts = line[1:].strip().split(',', 1)
                if len(parts) == 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    if k == 'sample_rate_scg_hz':
                        expected_scg = int(v)
                    elif k == 'sample_rate_ppg_hz':
                        expected_ppg = int(v)
            else:
                break
    return expected_scg, expected_ppg

def process_file(csv_path):
    """Parses a CSV file and calculates hardware timing error metrics."""
    file_name = os.path.basename(csv_path)
    
    result = {
        'file_name': file_name,
        'scg_expected_fs': 256,
        'scg_actual_fs': np.nan,
        'scg_mean_interval_ms': np.nan,
        'scg_interval_error_ms': np.nan,
        'scg_jitter_ms': np.nan,
        'scg_error_pct': np.nan,
        'ppg_expected_fs': 100,
        'ppg_actual_fs': np.nan,
        'ppg_mean_interval_ms': np.nan,
        'ppg_interval_error_ms': np.nan,
        'ppg_jitter_ms': np.nan,
        'ppg_error_pct': np.nan,
        'ibi_mean_error_ms': np.nan,
        'status': 'Success'
    }

    try:
        # 1. Parse Expected Rates
        scg_expected, ppg_expected = parse_metadata(csv_path)
        result['scg_expected_fs'] = scg_expected
        result['ppg_expected_fs'] = ppg_expected

        expected_scg_interval = 1000.0 / scg_expected
        expected_ppg_interval = 1000.0 / ppg_expected

        # 2. Load File
        df = pd.read_csv(csv_path, comment='#')

        # 3. Analyze SCG timing (Accelerometer)
        scg_df = df[df['x_g'].notna()]
        if len(scg_df) >= 2:
            scg_ts = scg_df['timestamp_ms'].to_numpy()
            scg_intervals = np.diff(scg_ts)
            
            scg_mean_int = np.mean(scg_intervals)
            scg_actual_fs = 1000.0 / scg_mean_int
            scg_int_err = scg_mean_int - expected_scg_interval
            scg_jitter = np.std(scg_intervals)
            scg_err_pct = abs(scg_actual_fs - scg_expected) / scg_expected * 100.0

            result['scg_actual_fs'] = scg_actual_fs
            result['scg_mean_interval_ms'] = scg_mean_int
            result['scg_interval_error_ms'] = scg_int_err
            result['scg_jitter_ms'] = scg_jitter
            result['scg_error_pct'] = scg_err_pct

        # 4. Analyze PPG timing (Photoplethysmography)
        ppg_df = df[df['ppg_raw'].notna()]
        if len(ppg_df) >= 2:
            ppg_ts = ppg_df['timestamp_ms'].to_numpy()
            ppg_intervals = np.diff(ppg_ts)
            
            ppg_mean_int = np.mean(ppg_intervals)
            ppg_actual_fs = 1000.0 / ppg_mean_int
            ppg_int_err = ppg_mean_int - expected_ppg_interval
            ppg_jitter = np.std(ppg_intervals)
            ppg_err_pct = abs(ppg_actual_fs - ppg_expected) / ppg_expected * 100.0

            result['ppg_actual_fs'] = ppg_actual_fs
            result['ppg_mean_interval_ms'] = ppg_mean_int
            result['ppg_interval_error_ms'] = ppg_int_err
            result['ppg_jitter_ms'] = ppg_jitter
            result['ppg_error_pct'] = ppg_err_pct

        # 5. Analyze Heart Beat Peak Detection Accuracy
        beat_df = df[df['beat_event'] == 1]
        if len(beat_df) >= 2:
            beat_ts = beat_df['timestamp_ms'].to_numpy()
            reported_ibi = beat_df['ibi_ms'].to_numpy()
            
            calc_ibi = np.diff(beat_ts)
            reported_ibi_aligned = reported_ibi[1:]
            
            valid_mask = ~np.isnan(reported_ibi_aligned)
            if np.any(valid_mask):
                errors = calc_ibi[valid_mask] - reported_ibi_aligned[valid_mask]
                result['ibi_mean_error_ms'] = np.mean(np.abs(errors))

    except Exception as e:
        result['status'] = 'Failed'
        
    return result

def main():
    print("=" * 80)
    print("           Batch Seismocardiography & PPG Hardware Timing Validator           ")
    print("=" * 80)

    # Search for all CSV files recursively in SUBJECT_Data
    search_pattern = os.path.join('SUBJECT_Data', '**', '*.csv')
    csv_files = sorted(glob.glob(search_pattern, recursive=True))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('.')]

    total_files = len(csv_files)
    if total_files == 0:
        print("Error: No CSV recording files found in 'SUBJECT_Data/'.")
        sys.exit(1)

    print(f"Recursively crawling 'SUBJECT_Data/'. Found {total_files} files.\n")

    all_results = []
    
    for idx, csv_path in enumerate(csv_files, 1):
        rel_path = os.path.relpath(csv_path, start='SUBJECT_Data')
        print(f"  [{idx}/{total_files}] Processing: {rel_path:<60} ... ", end='', flush=True)
        
        file_result = process_file(csv_path)
        all_results.append(file_result)
        
        if file_result['status'] == 'Success':
            print("\033[92m\033[1mSuccess\033[0m")
        else:
            print("\033[91m\033[1mFAILED\033[0m")

    # Filter out failures, convert to DataFrame
    valid_results = [r for r in all_results if r['status'] == 'Success']
    df_summary = pd.DataFrame(valid_results)
    
    # Drop internal status column
    if 'status' in df_summary.columns:
        df_summary = df_summary.drop(columns=['status'])

    # Ensure output directory exists
    output_dir = 'Data_Validation'
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, 'validation_summary.csv')
    df_summary.to_csv(output_csv, index=False)
    print(f"\nSummary table saved to: \033[96m\033[1m{output_csv}\033[0m")

    # Calculate detailed statistics for boxplots and terminal output
    scg_err = df_summary['scg_error_pct'].dropna()
    ppg_err = df_summary['ppg_error_pct'].dropna()
    scg_jit = df_summary['scg_jitter_ms'].dropna()
    ppg_jit = df_summary['ppg_jitter_ms'].dropna()

    def get_stats(series):
        if len(series) == 0:
            return {}
        return {
            'mean': np.mean(series),
            'std': np.std(series),
            'min': np.min(series),
            'q25': np.percentile(series, 25),
            'median': np.median(series),
            'q75': np.percentile(series, 75),
            'max': np.max(series)
        }

    scg_err_stats = get_stats(scg_err)
    ppg_err_stats = get_stats(ppg_err)
    scg_jit_stats = get_stats(scg_jit)
    ppg_jit_stats = get_stats(ppg_jit)

    # Print a structured timing statistical report to the terminal
    print("\n" + "\033[1m" + "="*80)
    print("                 HARDWARE SAMPLING ACCURACY STATISTICAL SUMMARY                 ")
    print("="*80 + "\033[0m")
    
    def print_stat_row(name, stats, unit="%"):
        if not stats:
            print(f"  {name:<25}: No data available")
            return
        print(f"  {name:<25}: \033[92m\033[1mMean={stats['mean']:.4f}{unit}\033[0m | \033[94m\033[1mSD={stats['std']:.4f}{unit}\033[0m")
        print(f"                             [Min={stats['min']:.4f} | Q1={stats['q25']:.4f} | \033[96m\033[1mMedian={stats['median']:.4f}\033[0m | Q3={stats['q75']:.4f} | Max={stats['max']:.4f}]")

    print_stat_row("SCG Frequency Error", scg_err_stats, "%")
    print_stat_row("PPG Frequency Error", ppg_err_stats, "%")
    print("-" * 80)
    print_stat_row("SCG Timing Jitter", scg_jit_stats, " ms")
    print_stat_row("PPG Timing Jitter", ppg_jit_stats, " ms")
    
    if 'ibi_mean_error_ms' in df_summary.columns:
        ibi_err = df_summary['ibi_mean_error_ms'].dropna()
        if len(ibi_err) > 0:
            ibi_stats = get_stats(ibi_err)
            print("-" * 80)
            print_stat_row("MCU Peak-Detect IBI MAE", ibi_stats, " ms")
            
    print("\033[1m" + "="*80 + "\033[0m\n")

    # Generate timing accuracy charts
    print("Generating timing accuracy visualization charts...")
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Hardware Sampling Accuracy & Jitter Analysis (Device Timestamps)", 
                 fontsize=14, fontweight='bold', color='#2c3e50')

    # Color definitions
    scg_color = '#1f77b4' # Accent Blue
    ppg_color = '#2ecc71' # Accent Green

    # --- PLOT 1: Sampling Frequency Error Percentage (%) ---
    ax = axs[0]
    errors_pct = [scg_err, ppg_err]
    
    bplot = ax.boxplot(errors_pct, vert=True, patch_artist=True)
    ax.set_xticklabels(['SCG (256 Hz)', 'PPG (100 Hz)'])
    
    colors = [scg_color, ppg_color]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax.set_title("Sampling Frequency Percentage Error Distribution", fontweight='bold', fontsize=11)
    ax.set_ylabel("Sampling Rate Error (%)")
    ax.set_yscale('log' if df_summary['scg_error_pct'].max() > 1.0 or df_summary['ppg_error_pct'].max() > 1.0 else 'linear')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Place stats text box on Plot 1 (Frequency Error)
    textstr1 = '\n'.join((
        r'$\bf{SCG\ (256\ Hz)\ Fs\ Error:}$',
        f"  Mean: {scg_err_stats['mean']:.4f}%",
        f"  Median: {scg_err_stats['median']:.4f}%",
        f"  SD: {scg_err_stats['std']:.4f}%",
        r'$\bf{PPG\ (100\ Hz)\ Fs\ Error:}$',
        f"  Mean: {ppg_err_stats['mean']:.4f}%",
        f"  Median: {ppg_err_stats['median']:.4f}%",
        f"  SD: {ppg_err_stats['std']:.4f}%"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#bdc3c7')
    ax.text(0.05, 0.95, textstr1, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    # --- PLOT 2: Timing Jitter Distribution (ms) ---
    ax = axs[1]
    jitter_data = [scg_jit, ppg_jit]
    
    bplot2 = ax.boxplot(jitter_data, vert=True, patch_artist=True)
    ax.set_xticklabels(['SCG (256 Hz)', 'PPG (100 Hz)'])
    
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax.set_title("Hardware Sample Timing Jitter Distribution", fontweight='bold', fontsize=11)
    ax.set_ylabel("Jitter Standard Deviation (ms)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Place stats text box on Plot 2 (Timing Jitter)
    textstr2 = '\n'.join((
        r'$\bf{SCG\ (256\ Hz)\ Timing\ Jitter:}$',
        f"  Mean: {scg_jit_stats['mean']:.4f} ms",
        f"  Median: {scg_jit_stats['median']:.4f} ms",
        f"  SD: {scg_jit_stats['std']:.4f} ms",
        r'$\bf{PPG\ (100\ Hz)\ Timing\ Jitter:}$',
        f"  Mean: {ppg_jit_stats['mean']:.4f} ms",
        f"  Median: {ppg_jit_stats['median']:.4f} ms",
        f"  SD: {ppg_jit_stats['std']:.4f} ms"
    ))
    ax.text(0.05, 0.95, textstr2, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Custom annotations for clarity
    ax.axhline(0.2915, color='#e74c3c', linestyle=':', linewidth=1.5, 
               label='Theoretical SCG Quantization Limit (0.29 ms)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    plt.tight_layout()
    
    # Save image
    output_png = os.path.join(output_dir, 'validation_summary.png')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Timing visualization graph saved to: \033[96m\033[1m{output_png}\033[0m")
    print("=" * 80)

if __name__ == "__main__":
    main()
