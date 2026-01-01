import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = '/Users/amansingh/Desktop/HDD/data_Q1_2025'

def analyze_failures():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} files.")

    # Pass 1: Identify failed drives
    failed_drives = set()
    failed_models = set()
    
    print("Pass 1: Identifying failed drives...")
    for idx, f in enumerate(csv_files):
        if idx % 10 == 0:
            print(f"Scanning file {idx+1}/{len(csv_files)}...")
        try:
            with open(f, 'r') as fp:
                # Skip header
                header = next(fp)
                # Find index of failure (usually 4th index -> 5th column)
                headers = header.strip().split(',')
                try:
                    fail_idx = headers.index('failure')
                    serial_idx = headers.index('serial_number')
                    model_idx = headers.index('model')
                except ValueError:
                    continue
                
                for line in fp:
                    parts = line.split(',')
                    if len(parts) > fail_idx and parts[fail_idx] == '1':
                        failed_drives.add(parts[serial_idx])
                        failed_models.add(parts[model_idx])
                        
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
            
    if not failed_drives:
        print("No failed drives found.")
        return

    print(f"Found {len(failed_drives)} failed drives: {failed_drives}")
    print(f"Models of failed drives: {failed_models}")

    # Pass 2: Collect data for failed drives
    print("Pass 2: Collecting data history...")
    drive_history = {serial: [] for serial in failed_drives}
    
    # Identify SMART columns (raw) from the first file
    first_df = pd.read_csv(csv_files[0], nrows=1)
    smart_cols = [c for c in first_df.columns if c.startswith('smart_') and c.endswith('_raw')]
    
    cols_to_read = ['date', 'serial_number'] + smart_cols
    
    for f in csv_files:
        try:
            # We must read serial_number to filter
            # To speed up, we might need to read chunks if files are huge, but 300k rows is okay for pandas
            df = pd.read_csv(f, usecols=cols_to_read)
            filtered = df[df['serial_number'].isin(failed_drives)]
            
            for _, row in filtered.iterrows():
                drive_history[row['serial_number']].append(row.to_dict())
                
        except Exception as e:
            print(f"Error processing {f} in pass 2: {e}")

    # Analysis
    print("Analyzing parameters...")
    
    # We want to find parameters that changed abnormally.
    # We will look at the deviation/change magnitude for each parameter per drive.
    
    param_scores = {col: 0.0 for col in smart_cols}
    
    for serial, rows in drive_history.items():
        if not rows:
            continue
        df_drive = pd.DataFrame(rows).sort_values('date')
        
        # Calculate statistics for each raw parameter
        for col in smart_cols:
            series = df_drive[col]
            if series.nunique() <= 1:
                continue # No change, skip
            
            # Metric: Range of values or Standard Deviation
            # Since raw values can be huge, we can look at relative change or normalized change
            # But the prompt asks for "raw parameters".
            # "Changed abnormally" usually implies a spike or drift.
            # Let's use the standard deviation as a proxy for activity/instability.
            # Or the difference between max and min.
            
            # Simple heuristic: Accumulate the (Max - Min) normalized by (Max + 1) to handle scale?
            # Or just count how many drives showed *any* change in this parameter.
            # And weigh by variance.
            
            # Let's try: Number of unique values > 1 implies change.
            # Score = Variance of the series (normalized?)
            
            # Problem: 'Seek Error Rate' might change a lot but be normal.
            # 'Reallocated Sectors' changing from 0 to 1 is very significant.
            
            # Let's count significant deviations.
            # Let's look at the "drift": abs(last - first).
            # If a parameter changes monotonically (like error counts), last - first is good.
            # If it fluctuates (like temperature), std dev is good.
            
            val_min = series.min()
            val_max = series.max()
            
            if val_min == val_max:
                continue
                
            # Normalize change by identifying if it's a known counter? 
            # We don't have domain knowledge coded in.
            
            # Let's assume parameters that change *at all* on failed drives are candidates.
            # We want the "5 most important".
            # Let's count how many failed drives had a change in this parameter.
            # And sum the normalized variance.
            
            # Normalized range: (max - min) / (max + 1) (strictly positive, 0-1ish?)
            # No, raw values can be large.
            
            # Let's just track the number of drives where this parameter changed.
            # Ties broken by total accumulated absolute change?
            
            param_scores[col] += 1 # Count drives with change

            # Another metric: Correlation with time?
            
    # Sort by number of drives affected
    sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select critical parameters including error counts and lifespan indicators
    candidate_params = [
        'smart_5_raw',   # Reallocated Sector Count
        'smart_187_raw', # Reported Uncorrectable Errors
        'smart_197_raw', # Current Pending Sector Count
        'smart_198_raw', # Offline Uncorrectable Sector Count
        'smart_1_raw',   # Read Error Rate
        'smart_188_raw', # Command Timeout
        'smart_192_raw', # Power-off Retract Count
        'smart_4_raw',   # Start/Stop Count
        'smart_7_raw',   # Seek Error Rate
        'smart_9_raw'    # Power On Hours (Control)
    ]
    
    print("\nDetailed stats for candidate critical parameters on failed drives:")
    
    results = []

    for col in candidate_params:
        if col not in smart_cols:
            continue
            
        final_values = []
        changes = []
        start_values = []
        
        count_changed = 0
        drive_count = 0
        
        for serial, rows in drive_history.items():
            if not rows:
                continue
            
            # Create DataFrame and ensure numeric
            df_drive = pd.DataFrame(rows).sort_values('date')
            
            # Coerce to numeric, turning non-parseable to NaN
            series = pd.to_numeric(df_drive[col], errors='coerce')
            
            # Drop NaNs for valid comparison
            valid_series = series.dropna()
            
            if valid_series.empty:
                continue
                
            val_start = valid_series.iloc[0]
            val_end = valid_series.iloc[-1]
            
            drive_count += 1
            
            # Check for change
            if val_start != val_end:
                count_changed += 1
                changes.append(val_end - val_start)
            
            final_values.append(val_end)
            start_values.append(val_start)

        if final_values:
            avg_final = np.mean(final_values)
            max_final = np.max(final_values)
            avg_change = np.mean(changes) if changes else 0
            max_change = np.max(changes) if changes else 0
            
            print(f"{col}: Analyzed {drive_count} drives. Changed in {count_changed}. Avg Change: {avg_change:.2f}, Max Change: {max_change}")
            
            results.append({
                'param': col,
                'changed_count': count_changed,
                'avg_final': avg_final,
                'max_final': max_final,
                'avg_change': avg_change
            })

    # Sort results by prevalence of change (excluding smart_9 if possible, but keeping for usage)
    sorted_res = sorted(results, key=lambda x: x['changed_count'], reverse=True)
    
    print("\nTop 5 Candidates:")
    for r in sorted_res[:10]:
        print(r)
    
    return sorted_res[:5]

if __name__ == "__main__":
    analyze_failures()
