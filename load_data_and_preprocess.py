import os
import glob
import pandas as pd

# Limit math library threads to reduce peak memory/CPU contention
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

# Columns needed
usecols = [
    'date', 'serial_number', 'model', 'failure',
    'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw'
]

# Stream all CSVs and aggregate per-serial statistics without loading everything into memory
csv_paths = sorted(glob.glob(os.path.join('data_Q1_2025', '*.csv')))

# Aggregation stores
survival_map = {}  # serial -> {start_date, end_date, event}
mean_sums = {}     # serial -> {sum_5, sum_9, count}
max_map = {}       # serial -> {max_187, max_197, max_198}

chunksize = 200_000
for path in csv_paths:
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        # Parse date
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')

        # Survival per chunk
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

        # Features per chunk
        # For means, accumulate sums and counts; for maxima, take max
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

# Build final DataFrames
survival_df = pd.DataFrame.from_dict(survival_map, orient='index')
survival_df.index.name = 'serial_number'
survival_df['duration'] = (survival_df['end_date'] - survival_df['start_date']).dt.days

means_df = pd.DataFrame.from_dict(mean_sums, orient='index')
means_df.index.name = 'serial_number'
for col in ['sum_5', 'sum_9', 'count']:
    if col not in means_df:
        means_df[col] = 0
means_df['smart_5_raw'] = means_df['sum_5'] / means_df['count'].replace(0, pd.NA)
means_df['smart_9_raw'] = means_df['sum_9'] / means_df['count'].replace(0, pd.NA)
means_df = means_df[['smart_5_raw', 'smart_9_raw']]

max_df = pd.DataFrame.from_dict(max_map, orient='index')
max_df.index.name = 'serial_number'

features = means_df.join(max_df, how='outer')
survival_df = survival_df.join(features, on='serial_number')

print({'n_serials': len(survival_df), 'n_features': features.shape[1]})

from lifelines import CoxPHFitter

cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)

# Prepare modeling dataframe: exclude datetime columns and keep only numeric covariates
covariate_cols = [
    'smart_5_raw',
    'smart_9_raw',
    'smart_187_raw',
    'smart_197_raw',
    'smart_198_raw',
]
model_df = survival_df[['duration', 'event'] + covariate_cols].copy()

# Drop rows with non-positive duration (no time at risk)
model_df = model_df[model_df['duration'] > 0]

# Lifelines expects numeric inputs; replace NaNs if present
model_df = model_df.astype({'duration': 'float64', 'event': 'float64'})
model_df[covariate_cols] = model_df[covariate_cols].astype('float64')
model_df = model_df.fillna(0.0)

# Drop zero-variance covariates to reduce multicollinearity
zero_var_cols = [c for c in covariate_cols if model_df[c].nunique(dropna=False) <= 1]
if zero_var_cols:
    covariate_cols = [c for c in covariate_cols if c not in zero_var_cols]
    model_df = model_df[['duration', 'event'] + covariate_cols]

cph.fit(model_df, duration_col='duration', event_col='event')
cph.print_summary()
import time
import psutil

end_time = time.strftime("%Y-%m-%d %H:%M:%S")
mem = psutil.virtual_memory()
used_gb = (mem.used / 1e9)
avail_gb = (mem.available / 1e9)

print("\n✅ Pipeline completed successfully!")
print(f"Timestamp: {end_time}")
print(f"Total drives processed: {len(survival_df):,}")
print(f"Features used: {len(covariate_cols)}")
print(f"Memory used: {used_gb:.2f} GB | Available: {avail_gb:.2f} GB\n")
