import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = '/Users/amansingh/Desktop/HDD/data_Q1_2025'

def visualize_failures():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} files.")

    # Pass 1: Identify failed drives (Quick scan)
    failed_drives = set()
    failed_models_map = {} # serial -> model
    
    print("Pass 1: Identifying failed drives...")
    for f in csv_files:
        try:
            with open(f, 'r') as fp:
                header = next(fp)
                headers = header.strip().split(',')
                fail_idx = headers.index('failure')
                serial_idx = headers.index('serial_number')
                model_idx = headers.index('model')
                
                for line in fp:
                    parts = line.split(',')
                    if len(parts) > fail_idx and parts[fail_idx] == '1':
                        serial = parts[serial_idx]
                        failed_drives.add(serial)
                        failed_models_map[serial] = parts[model_idx]
        except Exception:
            continue
            
    if not failed_drives:
        print("No failed drives found.")
        return

    print(f"Found {len(failed_drives)} failed drives.")

    # Target parameters for visualization
    target_params = [
        'smart_1_raw', 'smart_7_raw', 'smart_187_raw', 
        'smart_198_raw', 'smart_188_raw', 'smart_5_raw', 'smart_197_raw'
    ]
    
    # Pass 2: Collect time series for failed drives
    print("Pass 2: Collecting time series history for visualization...")
    drive_history = {serial: [] for serial in failed_drives}
    
    cols_to_read = ['date', 'serial_number'] + target_params
    
    for f in csv_files:
        try:
            # Using low_memory=False to avoid DtypeWarnings
            df = pd.read_csv(f, usecols=lambda x: x in cols_to_read)
            filtered = df[df['serial_number'].isin(failed_drives)]
            
            for _, row in filtered.iterrows():
                drive_history[row['serial_number']].append(row.to_dict())
        except Exception:
            continue

    # Plot 1: Prevalence Bar Chart
    print("Generating Prevalence Chart...")
    counts = {p: 0 for p in target_params}
    for serial, rows in drive_history.items():
        if not rows: continue
        df_drive = pd.DataFrame(rows)
        for p in target_params:
            if p in df_drive.columns:
                series = pd.to_numeric(df_drive[p], errors='coerce').dropna()
                if not series.empty and series.iloc[0] != series.iloc[-1]:
                    counts[p] += 1
    
    plt.figure(figsize=(12, 7))
    params_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0].replace('_raw', '').replace('smart_', 'SMART ') for x in params_sorted]
    values = [x[1] for x in params_sorted]
    
    plt.bar(labels, values, color='skyblue', edgecolor='navy')
    plt.title('Prevalence of Abnormal Changes in Failed Drives', fontsize=15)
    plt.ylabel('Number of Drives Affected', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('parameter_prevalence.png')
    print("Saved parameter_prevalence.png")

    # Plot 2: Time Series for sample drives
    print("Generating Sample Time Series for 'death spiral'...")
    # Select a few drives that have changes in multiple parameters
    sample_serials = []
    for serial, rows in drive_history.items():
        if len(rows) > 10: # Want some history
            sample_serials.append(serial)
        if len(sample_serials) >= 3: break
    
    for i, serial in enumerate(sample_serials):
        df_drive = pd.DataFrame(drive_history[serial]).sort_values('date')
        df_drive['date'] = pd.to_datetime(df_drive['date'])
        
        plt.figure(figsize=(14, 8))
        for p in target_params:
            if p in df_drive.columns:
                series = pd.to_numeric(df_drive[p], errors='coerce')
                if not series.dropna().empty and series.max() > 0:
                    # Normalize for visibility in one plot (0 to 1 scaling)
                    s_min = series.min()
                    s_max = series.max()
                    if s_max > s_min:
                        norm_series = (series - s_min) / (s_max - s_min)
                        plt.plot(df_drive['date'], norm_series, marker='o', label=f"{p} (max={s_max})")
        
        plt.title(f"Normalized SMART Parameter Evolution - Drive: {serial} ({failed_models_map[serial]})", fontsize=15)
        plt.xlabel('Date')
        plt.ylabel('Normalized Value (0 to 1)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'death_spiral_sample_{i+1}.png')
        print(f"Saved death_spiral_sample_{i+1}.png")

if __name__ == "__main__":
    visualize_failures()
