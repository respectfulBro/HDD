import os
import glob
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Limit math library threads to reduce peak memory/CPU contention
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

print("="*80)
print(f"MODEL-SPECIFIC FAILURE ANALYSIS: CT250MX500SSD1")
print(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Target model
TARGET_MODEL = 'CT250MX500SSD1'

# Use same columns as load_data_and_preprocess.py
usecols = [
    'date', 'serial_number', 'model', 'failure',
    'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw',
    'smart_1_raw', 'smart_7_raw', 'smart_196_raw', 'smart_194_raw'
]

# Get all CSV files
csv_paths = sorted(glob.glob(os.path.join('data_Q1_2025', '*.csv')))
print(f"\nFound {len(csv_paths)} CSV files to process")

# Aggregation stores (same as preprocessing script)
survival_map = {}  # serial -> {start_date, end_date, event}
mean_sums = {}     # serial -> {sum_5, sum_9, count}
max_map = {}       # serial -> {max_187, max_197, max_198}
first_last = {}    # first/last tracking for SMART features
temp_stats = {}    # temperature mean/std/max
binary_flags = {}  # uncorrectable, pending, offline uncorrectable binary indicators

chunksize = 200_000
total_devices = 0
failed_devices = 0

print(f"\nFiltering data for model: {TARGET_MODEL}")

for i, path in enumerate(csv_paths, 1):
    print(f"\rProcessing file {i}/{len(csv_paths)}: {os.path.basename(path)}", end='', flush=True)
    
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        # Filter for target model ONLY
        chunk = chunk[chunk['model'] == TARGET_MODEL].copy()
        
        if len(chunk) == 0:
            continue
        
        # Parse date
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')

        # =====================================================================
        # 1. SURVIVAL STATS
        # =====================================================================
        surv = chunk.groupby('serial_number').agg(
            start_date=('date', 'min'),
            end_date=('date', 'max'),
            event=('failure', 'max'),
        )
        for serial, row in surv.iterrows():
            if serial not in survival_map:
                survival_map[serial] = {
                    'start_date': row['start_date'],
                    'end_date': row['end_date'],
                    'event': row['event'],
                }
            else:
                agg = survival_map[serial]
                if row['start_date'] < agg['start_date']:
                    agg['start_date'] = row['start_date']
                if row['end_date'] > agg['end_date']:
                    agg['end_date'] = row['end_date']
                if row['event'] > agg['event']:
                    agg['event'] = row['event']

        # =====================================================================
        # 2. MEANS
        # =====================================================================
        f_mean = chunk.groupby('serial_number').agg({
            'smart_5_raw': 'sum',
            'smart_9_raw': 'sum',
        })
        f_count = chunk.groupby('serial_number').agg(count=('smart_5_raw', 'size'))
        f_mean = f_mean.join(f_count)

        for serial, row in f_mean.iterrows():
            if serial not in mean_sums:
                mean_sums[serial] = {
                    'sum_5': pd.to_numeric(row['smart_5_raw'], errors='coerce'),
                    'sum_9': pd.to_numeric(row['smart_9_raw'], errors='coerce'),
                    'count': int(row['count']),
                }
            else:
                agg = mean_sums[serial]
                agg['sum_5'] = (agg['sum_5'] if pd.notna(agg['sum_5']) else 0) + pd.to_numeric(row['smart_5_raw'], errors='coerce')
                agg['sum_9'] = (agg['sum_9'] if pd.notna(agg['sum_9']) else 0) + pd.to_numeric(row['smart_9_raw'], errors='coerce')
                agg['count'] += int(row['count'])

        # =====================================================================
        # 3. MAX VALUES
        # =====================================================================
        f_max = chunk.groupby('serial_number').agg({
            'smart_187_raw': 'max',
            'smart_197_raw': 'max',
            'smart_198_raw': 'max',
        })
        for serial, row in f_max.iterrows():
            if serial not in max_map:
                max_map[serial] = {
                    'smart_187_raw': row['smart_187_raw'],
                    'smart_197_raw': row['smart_197_raw'],
                    'smart_198_raw': row['smart_198_raw'],
                }
            else:
                agg = max_map[serial]
                for k in ['smart_187_raw', 'smart_197_raw', 'smart_198_raw']:
                    if pd.isna(agg[k]) or (pd.notna(row[k]) and row[k] > agg[k]):
                        agg[k] = row[k]

        # =====================================================================
        # 4. NEW FEATURE BLOCKS
        # =====================================================================

        # A. FIRST/LAST tracking for SMART progression
        fl_metrics = ['smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_196_raw']
        chunk_sorted = chunk.sort_values(['serial_number', 'date'])

        first_vals = chunk_sorted.groupby('serial_number')[fl_metrics].first()
        last_vals = chunk_sorted.groupby('serial_number')[fl_metrics].last()

        for serial in first_vals.index:
            if serial not in first_last:
                first_last[serial] = {}
                for col in fl_metrics:
                    first_last[serial][f'first_{col}'] = first_vals.loc[serial, col]
                    first_last[serial][f'last_{col}'] = last_vals.loc[serial, col]
            else:
                for col in fl_metrics:
                    if first_vals.loc[serial, col] < first_last[serial][f'first_{col}']:
                        first_last[serial][f'first_{col}'] = first_vals.loc[serial, col]
                    if last_vals.loc[serial, col] > first_last[serial][f'last_{col}']:
                        first_last[serial][f'last_{col}'] = last_vals.loc[serial, col]

        # B. Temperature stats (mean, std, max)
        temp_group = chunk.groupby('serial_number')['smart_194_raw'].agg(['sum', 'max', 'count'])
        temp_group['sumsq'] = chunk.groupby('serial_number')['smart_194_raw'].apply(lambda x: (x**2).sum())

        for serial, row in temp_group.iterrows():
            if serial not in temp_stats:
                temp_stats[serial] = {
                    'sum': row['sum'],
                    'sumsq': row['sumsq'],
                    'max': row['max'],
                    'count': row['count'],
                }
            else:
                t = temp_stats[serial]
                t['sum'] += row['sum']
                t['sumsq'] += row['sumsq']
                t['count'] += row['count']
                if row['max'] > t['max']:
                    t['max'] = row['max']

        # C. Binary early failure flags
        bin_group = chunk.groupby('serial_number').agg({
            'smart_187_raw': 'max',
            'smart_197_raw': 'max',
            'smart_198_raw': 'max',
        })

        for serial, row in bin_group.iterrows():
            if serial not in binary_flags:
                binary_flags[serial] = {
                    'uncorrectable_present': int(row['smart_187_raw'] > 0),
                    'pending_present': int(row['smart_197_raw'] > 0),
                    'offline_uncorrectable_present': int(row['smart_198_raw'] > 0),
                }
            else:
                f = binary_flags[serial]
                f['uncorrectable_present'] |= int(row['smart_187_raw'] > 0)
                f['pending_present'] |= int(row['smart_197_raw'] > 0)
                f['offline_uncorrectable_present'] |= int(row['smart_198_raw'] > 0)

print("\n\n" + "="*80)
print("DATA PROCESSING COMPLETE - Building Feature Dataset")
print("="*80)

# =====================================================================
# FINAL AGGREGATION
# =====================================================================

# Survival table
survival_df = pd.DataFrame.from_dict(survival_map, orient='index')
survival_df.index.name = 'serial_number'
survival_df['duration'] = (survival_df['end_date'] - survival_df['start_date']).dt.days

# Means
means_df = pd.DataFrame.from_dict(mean_sums, orient='index')
means_df.index.name = 'serial_number'
for col in ['sum_5', 'sum_9', 'count']:
    if col not in means_df:
        means_df[col] = 0
means_df['smart_5_raw'] = means_df['sum_5'] / means_df['count'].replace(0, pd.NA)
means_df['smart_9_raw'] = means_df['sum_9'] / means_df['count'].replace(0, pd.NA)
means_df = means_df[['smart_5_raw', 'smart_9_raw']]

# Max
max_df = pd.DataFrame.from_dict(max_map, orient='index')
max_df.index.name = 'serial_number'

# Temperature DF
temp_df = pd.DataFrame.from_dict(temp_stats, orient='index')
temp_df.index.name = 'serial_number'
temp_df['temp_mean'] = temp_df['sum'] / temp_df['count']
temp_df['temp_std'] = ((temp_df['sumsq'] / temp_df['count']) - temp_df['temp_mean']**2).clip(lower=0).pow(0.5)
temp_df = temp_df.rename(columns={'max': 'temp_max'})
temp_df = temp_df[['temp_mean', 'temp_max', 'temp_std']]

# First/Last DF + deltas
fl_df = pd.DataFrame.from_dict(first_last, orient='index')
fl_df.index.name = 'serial_number'
for col in ['smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_196_raw']:
    fl_df[f'delta_{col}'] = fl_df[f'last_{col}'] - fl_df[f'first_{col}']

# Slopes: delta / duration
for col in ['smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_196_raw']:
    fl_df[f'slope_{col}'] = fl_df[f'delta_{col}'] / survival_df['duration']

# Binary flags
bin_df = pd.DataFrame.from_dict(binary_flags, orient='index')
bin_df.index.name = 'serial_number'

# Combine features
features = (
    means_df
    .join(max_df, how='outer')
    .join(temp_df, how='outer')
    .join(fl_df, how='outer')
    .join(bin_df, how='outer')
)

survival_df = survival_df.join(features, on='serial_number')

# Age transforms
survival_df['age_years'] = survival_df['duration'] / 365
survival_df['log_age'] = np.log1p(survival_df['duration'])

total_devices = len(survival_df)
failed_devices = survival_df['event'].sum()

print(f"\n📊 DATASET SUMMARY FOR {TARGET_MODEL}")
print(f"   Total devices: {total_devices:,}")
print(f"   Failed devices: {failed_devices:,}")
print(f"   Failure rate: {(failed_devices/total_devices*100):.2f}%")
print(f"   Total features: {features.shape[1]}")
print(f"   Date range: {survival_df['start_date'].min()} to {survival_df['end_date'].max()}")

# =====================================================================
# FAILURE PREDICTION MODELING
# =====================================================================
print("\n" + "="*80)
print("FAILURE ANALYSIS")
print("="*80)

if failed_devices == 0:
    print(f"\n🎉 EXCELLENT NEWS!")
    print(f"   No failures detected for {TARGET_MODEL} in Q1 2025 data!")
    print(f"   This model shows exceptional reliability with 0% failure rate.")
    print(f"   Total devices monitored: {total_devices:,}")
    print(f"   Total observation days: {survival_df['duration'].sum():,.0f}")
    
    # Since we can't build a failure model, let's analyze health indicators
    print("\n" + "="*80)
    print("HEALTH INDICATOR ANALYSIS")
    print("="*80)
    
    # Prepare analysis dataframe
    covariate_cols = [
        'smart_5_raw', 
        'smart_9_raw',
        'smart_187_raw',
        'smart_197_raw',
        'smart_198_raw',
        'temp_mean', 'temp_max', 'temp_std',
        'delta_smart_5_raw', 'delta_smart_196_raw', 'delta_smart_197_raw',
        'slope_smart_5_raw', 'slope_smart_196_raw', 'slope_smart_197_raw',
        'pending_present',
        'age_years', 'log_age'
    ]
    
    # Filter to available columns
    available_cols = [c for c in covariate_cols if c in survival_df.columns]
    analysis_df = survival_df[available_cols].copy().fillna(0.0)
    
    print(f"\n📊 SMART PARAMETER STATISTICS:")
    print(f"\n{'-'*80}")
    
    # Critical SMART parameters
    critical_params = {
        'smart_5_raw': 'Reallocated Sectors',
        'smart_9_raw': 'Power-On Hours',
        'smart_187_raw': 'Uncorrectable Errors',
        'smart_197_raw': 'Current Pending Sectors',
        'smart_198_raw': 'Offline Uncorrectable',
        'temp_mean': 'Average Temperature (°C)',
        'temp_max': 'Max Temperature (°C)',
    }
    
    for param, description in critical_params.items():
        if param in analysis_df.columns:
            values = analysis_df[param]
            print(f"\n{description} ({param}):")
            print(f"   Mean: {values.mean():.2f}")
            print(f"   Median: {values.median():.2f}")
            print(f"   Min: {values.min():.2f}")
            print(f"   Max: {values.max():.2f}")
            print(f"   Std Dev: {values.std():.2f}")
            
            # Flag concerning values
            if 'smart_5' in param and values.max() > 0:
                print(f"   ⚠️  {(values > 0).sum()} devices have reallocated sectors")
            elif 'smart_187' in param and values.max() > 0:
                print(f"   ⚠️  {(values > 0).sum()} devices have uncorrectable errors")
            elif 'smart_197' in param and values.max() > 0:
                print(f"   ⚠️  {(values > 0).sum()} devices have pending sectors")
            elif 'temp_max' in param and values.max() > 60:
                print(f"   ⚠️  {(values > 60).sum()} devices exceeded 60°C")
    
    # Identify devices with concerning metrics
    print("\n" + "="*80)
    print("DEVICES REQUIRING ATTENTION")
    print("="*80)
    
    concern_conditions = []
    if 'smart_5_raw' in analysis_df.columns:
        concern_conditions.append(analysis_df['smart_5_raw'] > 0)
    if 'smart_197_raw' in analysis_df.columns:
        concern_conditions.append(analysis_df['smart_197_raw'] > 0)
    if 'temp_max' in analysis_df.columns:
        concern_conditions.append(analysis_df['temp_max'] > 60)
    
    if concern_conditions:
        concerning_mask = pd.Series(False, index=analysis_df.index)
        for condition in concern_conditions:
            concerning_mask |= condition
        
        concerning_devices = survival_df[concerning_mask]
        
        if len(concerning_devices) > 0:
            print(f"\n⚠️  {len(concerning_devices)} devices ({len(concerning_devices)/len(survival_df)*100:.1f}%) show concerning metrics:")
            print(f"\nTop 20 devices to monitor closely:")
            
            # Create risk score based on available metrics
            risk_score = pd.Series(0.0, index=analysis_df.index)
            if 'smart_5_raw' in analysis_df.columns:
                risk_score += analysis_df['smart_5_raw'] * 10
            if 'smart_197_raw' in analysis_df.columns:
                risk_score += analysis_df['smart_197_raw'] * 5
            if 'temp_max' in analysis_df.columns:
                risk_score += (analysis_df['temp_max'] - 50).clip(lower=0)
            
            survival_df['health_risk_score'] = risk_score
            
            high_risk = survival_df.nlargest(20, 'health_risk_score')[[
                'duration', 'health_risk_score'] + 
                [c for c in ['smart_5_raw', 'smart_197_raw', 'smart_198_raw', 'temp_max'] if c in survival_df.columns]
            ]
            col_mapping = {
                'duration': 'Age (days)',
                'health_risk_score': 'Risk Score',
                'smart_5_raw': 'Reallocated',
                'smart_197_raw': 'Pending',
                'smart_198_raw': 'Offline Uncorr',
                'temp_max': 'Max Temp'
            }
            high_risk = high_risk.rename(columns=col_mapping)
            print(high_risk.to_string())
        else:
            print("\n✅ All devices show healthy metrics - no concerning values detected!")
    
    # =====================================================================
    # RELIABILITY METRICS
    # =====================================================================
    print("\n" + "="*80)
    print("RELIABILITY METRICS")
    print("="*80)
    
    total_device_days = survival_df['duration'].sum()
    total_device_years = total_device_days / 365.25
    
    print(f"\n📈 Operational Statistics:")
    print(f"   Total device-days: {total_device_days:,.0f}")
    print(f"   Total device-years: {total_device_years:,.1f}")
    print(f"   Failures observed: {failed_devices}")
    print(f"   Failure rate: 0.00%")
    print(f"   Mean Time Between Failures (MTBF): Unable to calculate (no failures)")
    print(f"   AFR (Annualized Failure Rate): 0.00%")
    
    print(f"\n⏱️  Observation Period per Device:")
    print(f"   Mean: {survival_df['duration'].mean():.1f} days ({survival_df['duration'].mean()/30:.1f} months)")
    print(f"   Median: {survival_df['duration'].median():.1f} days")
    print(f"   Min: {survival_df['duration'].min():.1f} days")
    print(f"   Max: {survival_df['duration'].max():.1f} days")
    
    # =====================================================================
    # VISUAL ANALYSIS
    # =====================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*80)
    
    # Create output directory for plots
    os.makedirs('analysis_results', exist_ok=True)
    
    # Plot 1: SMART parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_params = [
        ('smart_5_raw', 'Reallocated Sectors Count', 'red'),
        ('smart_9_raw', 'Power-On Hours', 'blue'),
        ('smart_197_raw', 'Current Pending Sectors', 'orange'),
        ('temp_mean', 'Average Temperature (°C)', 'green'),
        ('temp_max', 'Maximum Temperature (°C)', 'darkred'),
        ('age_years', 'Device Age (years)', 'purple'),
    ]
    
    for idx, (param, label, color) in enumerate(plot_params):
        if param in analysis_df.columns and idx < len(axes):
            data = analysis_df[param].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) > 0:
                axes[idx].hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')
                axes[idx].axvline(data.median(), color='red', linestyle='--', linewidth=2, 
                                label=f'Median: {data.median():.1f}')
                axes[idx].set_xlabel(label, fontsize=10)
                axes[idx].set_ylabel('Count', fontsize=10)
                axes[idx].set_title(label, fontsize=11, fontweight='bold')
                axes[idx].legend(fontsize=8)
                axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Health Metrics Distribution for {TARGET_MODEL}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('analysis_results/health_metrics_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_results/health_metrics_distribution.png")
    
    # Plot 2: Age vs Key SMART Parameters
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if 'smart_5_raw' in analysis_df.columns:
        axes[0].scatter(survival_df['age_years'], analysis_df['smart_5_raw'], alpha=0.5, s=20)
        axes[0].set_xlabel('Device Age (years)', fontsize=11)
        axes[0].set_ylabel('Reallocated Sectors', fontsize=11)
        axes[0].set_title('Age vs Reallocated Sectors', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    if 'smart_197_raw' in analysis_df.columns:
        axes[1].scatter(survival_df['age_years'], analysis_df['smart_197_raw'], alpha=0.5, s=20, color='orange')
        axes[1].set_xlabel('Device Age (years)', fontsize=11)
        axes[1].set_ylabel('Pending Sectors', fontsize=11)
        axes[1].set_title('Age vs Pending Sectors', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    if 'temp_max' in analysis_df.columns:
        axes[2].scatter(survival_df['age_years'], analysis_df['temp_max'], alpha=0.5, s=20, color='red')
        axes[2].set_xlabel('Device Age (years)', fontsize=11)
        axes[2].set_ylabel('Max Temperature (°C)', fontsize=11)
        axes[2].set_title('Age vs Max Temperature', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_results/age_vs_smart_parameters.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_results/age_vs_smart_parameters.png")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "="*80)
    print("SAVING ANALYSIS RESULTS")
    print("="*80)
    
    # Save detailed results
    output_df = survival_df[['start_date', 'end_date', 'duration', 'event'] + available_cols].copy()
    if 'health_risk_score' in survival_df.columns:
        output_df['health_risk_score'] = survival_df['health_risk_score']
    output_df.to_csv(f'analysis_results/{TARGET_MODEL}_health_analysis.csv')
    print(f"✓ Saved: analysis_results/{TARGET_MODEL}_health_analysis.csv")
    
    # Save devices to monitor
    if 'health_risk_score' in survival_df.columns and survival_df['health_risk_score'].max() > 0:
        monitor_devices = survival_df.nlargest(50, 'health_risk_score')[[
            'duration', 'health_risk_score'] + 
            [c for c in ['smart_5_raw', 'smart_197_raw', 'smart_198_raw', 'temp_max', 'smart_9_raw'] 
             if c in survival_df.columns]
        ]
        monitor_devices.to_csv(f'analysis_results/{TARGET_MODEL}_devices_to_monitor.csv')
        print(f"✓ Saved: analysis_results/{TARGET_MODEL}_devices_to_monitor.csv")
    
    # Save summary statistics
    summary_stats = analysis_df.describe().T
    summary_stats.to_csv(f'analysis_results/{TARGET_MODEL}_summary_statistics.csv')
    print(f"✓ Saved: analysis_results/{TARGET_MODEL}_summary_statistics.csv")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📋 SUMMARY FOR {TARGET_MODEL}")
    print(f"   • Total devices analyzed: {total_devices:,}")
    print(f"   • Devices that failed: {failed_devices} (0.00%)")
    print(f"   • Total observation time: {total_device_years:,.1f} device-years")
    print(f"   • This model shows EXCEPTIONAL reliability")
    
    print(f"\n🏆 CONCLUSION:")
    print(f"   The {TARGET_MODEL} shows NO FAILURES in Q1 2025 dataset.")
    print(f"   With {total_device_years:,.1f} device-years of observation and 0% failure rate,")
    print(f"   this model demonstrates exceptional reliability and durability.")
    print(f"   ")
    print(f"   Based on industry standards, SSDs typically have:")
    print(f"   - Consumer SSDs: 0.5%-2% annual failure rate")
    print(f"   - This model: 0.00% in observed period")
    print(f"   ")
    print(f"   Expected lifespan: Unable to predict from failure data (no failures observed)")
    print(f"   Recommendation: Continue monitoring; device appears very reliable")
    
else:
    # Original modeling code for when there are failures
    from lifelines import CoxPHFitter
    
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    
    # ... (keep the original modeling code)
    
print(f"\n📁 Output files saved to: ./analysis_results/")

end_time = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"\n✅ Analysis completed at: {end_time}")
print("="*80)

