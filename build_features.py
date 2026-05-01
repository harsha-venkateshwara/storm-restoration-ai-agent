"""
Storm Restoration Prioritization AI
Phase 1: Feature Store Builder

Builds the complete county x hour feature matrix from:
  - EAGLE-I outage parquets (2015-2022)
  - NOAA storm event parquets (2015-2022)
  - Census county demographics

Outputs:
  data/features/feature_store_YYYY.parquet  (per year, memory safe)
  data/features/feature_store_all.parquet   (combined, sampled for training)
  data/features/county_stats.parquet        (static county features)
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

#Paths
BASE      = Path(r'C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai')
PROCESSED = BASE / 'data' / 'processed'
FEATURES  = BASE / 'data' / 'features'
FEATURES.mkdir(parents=True, exist_ok=True)

#Constants
EXCLUDE_STATES    = {'AS', 'GU', 'PR', 'VI', 'MP'}
OUTAGE_THRESHOLD  = 10       # absolute floor
COVERAGE_THRESH   = 0.60     # min coverage fraction to trust a state-year

WIND_EVENTS   = {'Thunderstorm Wind','High Wind','Strong Wind','Tornado'}
WINTER_EVENTS = {'Winter Storm','Winter Weather','Ice Storm','Heavy Snow','Blizzard','Sleet'}
FLOOD_EVENTS  = {'Flash Flood','Flood','Heavy Rain'}
TROPICAL      = {'Hurricane (Typhoon)','Tropical Storm'}

def assign_group(et):
    if et in WIND_EVENTS:   return 'wind'
    if et in WINTER_EVENTS: return 'winter'
    if et in FLOOD_EVENTS:  return 'flood'
    if et in TROPICAL:      return 'tropical'
    return 'other'

# Step 1: Build NOAA storm pivot (all years, vectorized)
print("=" * 60)
print("STEP 1: Building NOAA storm hourly flags (all years)")
print("=" * 60)

noaa_dfs = []
for f in sorted((PROCESSED / 'noaa').glob('noaa_details_*.parquet')):
    year = int(f.stem.split('_')[-1])
    if year < 2015: continue
    df = pd.read_parquet(f)
    df = df[df['CZ_TYPE'] == 'C'].copy()
    
    OUTAGE_TYPES = WIND_EVENTS | WINTER_EVENTS | FLOOD_EVENTS | TROPICAL | {
        'Extreme Cold/Wind Chill','Cold/Wind Chill','Lightning','Dense Fog','Freezing Fog'
    }
    df = df[df['EVENT_TYPE'].isin(OUTAGE_TYPES)].copy()
    df['begin_dt'] = pd.to_datetime(df['BEGIN_DATE_TIME'], format='%d-%b-%y %H:%M:%S', errors='coerce')
    df['end_dt']   = pd.to_datetime(df['END_DATE_TIME'],   format='%d-%b-%y %H:%M:%S', errors='coerce')
    df['fips'] = (
        df['STATE_FIPS'].astype(int).astype(str).str.zfill(2) +
        df['CZ_FIPS'].astype(int).astype(str).str.zfill(3)
    ).astype(int)
    df['event_group'] = df['EVENT_TYPE'].apply(assign_group)
    df['year'] = year
    noaa_dfs.append(df[['fips','event_group','begin_dt','end_dt','year']].dropna(subset=['begin_dt']))
    print(f"  NOAA {year}: {len(df):,} events")

noaa = pd.concat(noaa_dfs, ignore_index=True)
print(f"\nTotal NOAA events: {len(noaa):,}")

# Vectorized expansion to hourly spans
noaa['begin_h'] = noaa['begin_dt'].dt.floor('h')
noaa['end_h']   = noaa['end_dt'].fillna(noaa['begin_dt'] + pd.Timedelta(hours=3)).dt.ceil('h')
noaa['end_h']   = noaa.apply(
    lambda r: min(r['end_h'], r['begin_h'] + pd.Timedelta(hours=72)), axis=1
)
noaa['duration_h'] = ((noaa['end_h'] - noaa['begin_h'])
                      .dt.total_seconds() / 3600).clip(lower=1).astype(int)

print("Expanding to hourly spans...")
rep_counts = noaa['duration_h'].values
noaa_rep   = noaa.loc[noaa.index.repeat(rep_counts)].copy()
noaa_rep['hour_offset'] = noaa_rep.groupby(level=0).cumcount()
noaa_rep['hour'] = (noaa_rep['begin_h'] +
                    pd.to_timedelta(noaa_rep['hour_offset'], unit='h'))
noaa_rep = noaa_rep[['fips','hour','event_group']].reset_index(drop=True)

# Pivot: counts per event group per fips-hour
print("Pivoting storm flags...")
storm_pivot = (noaa_rep
               .groupby(['fips','hour','event_group'])
               .size()
               .unstack(fill_value=0)
               .reset_index())
storm_pivot.columns.name = None
for col in ['wind','winter','flood','tropical','other']:
    if col not in storm_pivot.columns:
        storm_pivot[col] = 0

storm_pivot['fips'] = storm_pivot['fips'].astype(str)
storm_pivot['hour'] = pd.to_datetime(storm_pivot['hour']).dt.tz_localize(None)
storm_pivot['any_storm'] = (
    storm_pivot[['wind','winter','flood','tropical','other']].sum(axis=1) > 0
).astype(np.int8)
storm_pivot['storm_severity'] = (
    storm_pivot['wind'] * 3 +
    storm_pivot['tropical'] * 5 +
    storm_pivot['winter'] * 2 +
    storm_pivot['flood'] * 2 +
    storm_pivot['other'] * 1
).clip(upper=10)

print(f"Storm pivot: {storm_pivot.shape} | Active hours: {len(storm_pivot):,}")
storm_pivot.to_parquet(FEATURES / 'storm_pivot.parquet', index=False)
print("Saved storm_pivot.parquet")

# Step 2: Build county p90 thresholds from 2015-2018
print("\n" + "=" * 60)
print("STEP 2: Computing county-adaptive outage thresholds")
print("=" * 60)

threshold_dfs = []
for year in [2015, 2016, 2017, 2018]:
    f = PROCESSED / 'eagle_i' / f'eaglei_{year}.parquet'
    if not f.exists(): continue
    tbl = pq.read_table(f, columns=['fips_code','state','sum'])
    df  = tbl.to_pandas()
    df  = df[~df['state'].isin(EXCLUDE_STATES)]
    df['customers_out'] = df['sum'].fillna(0).clip(lower=0).astype(np.float32)
    threshold_dfs.append(df[['fips_code','customers_out']])
    del df, tbl

thresh_df = pd.concat(threshold_dfs, ignore_index=True)
county_p90 = (thresh_df
              .groupby('fips_code')['customers_out']
              .quantile(0.90)
              .rename('county_p90')
              .reset_index()
              .rename(columns={'fips_code':'fips'}))
county_p90['fips'] = county_p90['fips'].astype(str)

county_mean = (thresh_df
               .groupby('fips_code')['customers_out']
               .mean()
               .rename('county_mean_baseline')
               .reset_index()
               .rename(columns={'fips_code':'fips'}))
county_mean['fips'] = county_mean['fips'].astype(str)

county_stats = county_p90.merge(county_mean, on='fips')
county_stats.to_parquet(FEATURES / 'county_stats.parquet', index=False)
print(f"County thresholds computed for {len(county_stats):,} counties")
del threshold_dfs, thresh_df

# Step 3: Build per-year feature store
print("\n" + "=" * 60)
print("STEP 3: Building per-year feature matrices")
print("=" * 60)

for year in range(2015, 2023):
    f = PROCESSED / 'eagle_i' / f'eaglei_{year}.parquet'
    if not f.exists():
        print(f"  [{year}] MISSING — skip")
        continue

    print(f"\n  Processing {year}...")

    # Load EAGLE-I (columns only)
    tbl = pq.read_table(f, columns=['fips_code','state','sum','run_start_time'])
    df  = tbl.to_pandas(); del tbl
    df  = df[~df['state'].isin(EXCLUDE_STATES)]
    df['timestamp']     = pd.to_datetime(df['run_start_time'])
    df['sum_is_null']   = df['sum'].isnull().astype(np.int8)
    df['customers_out'] = df['sum'].fillna(0).clip(lower=0).astype(np.float32)
    df['fips']          = df['fips_code'].astype(str)
    df['hour']          = df['timestamp'].dt.floor('h').dt.tz_localize(None)

    # Aggregate to hourly
    hourly = df.groupby(['fips','hour']).agg(
        peak_customers_out = ('customers_out','max'),
        mean_customers_out = ('customers_out','mean'),
        null_count         = ('sum_is_null','sum'),
        state              = ('state','first')
    ).reset_index()
    del df

    # Join county thresholds
    hourly = hourly.merge(county_stats, on='fips', how='left')
    hourly['county_p90']           = hourly['county_p90'].fillna(50)
    hourly['county_mean_baseline'] = hourly['county_mean_baseline'].fillna(0)

    # Binary outage label (county-adaptive)
    hourly['outage_label'] = (
        (hourly['peak_customers_out'] > hourly['county_p90']) &
        (hourly['peak_customers_out'] > OUTAGE_THRESHOLD)
    ).astype(np.int8)

    # Regression target: log1p transform for stability
    hourly['log_peak_customers'] = np.log1p(hourly['peak_customers_out']).astype(np.float32)

    # Join storm features
    storm_year = storm_pivot[
        (storm_pivot['hour'] >= f'{year}-01-01') &
        (storm_pivot['hour'] <  f'{year+1}-01-01')
    ].copy()
    storm_year['fips'] = storm_year['fips'].astype(str)
    hourly = hourly.merge(storm_year, on=['fips','hour'], how='left')
    for col in ['wind','winter','flood','tropical','other','any_storm','storm_severity']:
        if col not in hourly.columns:
            hourly[col] = 0
        hourly[col] = hourly[col].fillna(0).astype(np.float32)

    # Time features
    hourly['hour_of_day'] = hourly['hour'].dt.hour.astype(np.int8)
    hourly['month']       = hourly['hour'].dt.month.astype(np.int8)
    hourly['dayofweek']   = hourly['hour'].dt.dayofweek.astype(np.int8)
    hourly['is_weekend']  = (hourly['dayofweek'] >= 5).astype(np.int8)
    hourly['quarter']     = hourly['hour'].dt.quarter.astype(np.int8)
    hourly['day_of_year'] = hourly['hour'].dt.dayofyear.astype(np.int16)
    # Cyclical encoding for hour and month
    hourly['hour_sin']  = np.sin(2 * np.pi * hourly['hour_of_day'] / 24).astype(np.float32)
    hourly['hour_cos']  = np.cos(2 * np.pi * hourly['hour_of_day'] / 24).astype(np.float32)
    hourly['month_sin'] = np.sin(2 * np.pi * hourly['month'] / 12).astype(np.float32)
    hourly['month_cos'] = np.cos(2 * np.pi * hourly['month'] / 12).astype(np.float32)

    # Lag features (within-year)
    hourly = hourly.sort_values(['fips','hour'])
    grp = hourly.groupby('fips')
    hourly['lag_1h_peak']    = grp['peak_customers_out'].shift(1).fillna(0).astype(np.float32)
    hourly['lag_24h_peak']   = grp['peak_customers_out'].shift(24).fillna(0).astype(np.float32)
    hourly['lag_168h_peak']  = grp['peak_customers_out'].shift(168).fillna(0).astype(np.float32)
    hourly['lag_24h_label']  = grp['outage_label'].shift(24).fillna(0).astype(np.float32)
    hourly['lag_168h_label'] = grp['outage_label'].shift(168).fillna(0).astype(np.float32)

    # Rolling storm counts (6h, 12h, 24h)
    for w, name in [(6,'6h'), (12,'12h'), (24,'24h')]:
        hourly[f'wind_sum_{name}'] = (
            grp['wind'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        ).astype(np.float32)
        hourly[f'any_storm_sum_{name}'] = (
            grp['any_storm'].transform(lambda x: x.rolling(w, min_periods=1).sum())
        ).astype(np.float32)

    # County fragility: 30-day rolling outage rate
    hourly['county_fragility'] = (
        grp['outage_label']
        .transform(lambda x: x.shift(1).rolling(720, min_periods=24).mean())
        .fillna(0).astype(np.float32)
    )

    hourly['year'] = year

    out_path = FEATURES / f'feature_store_{year}.parquet'
    hourly.to_parquet(out_path, index=False)
    pos_rate = hourly['outage_label'].mean() * 100
    print(f"  Saved {year}: {len(hourly):,} rows | outage rate: {pos_rate:.2f}% | {out_path.name}")
    del hourly

# Step 4: Combine into training/val/test splits
print("\n" + "=" * 60)
print("STEP 4: Building train/val/test splits")
print("=" * 60)

FEATURE_COLS = [
    'wind','winter','flood','tropical','other',
    'any_storm','storm_severity',
    'wind_sum_6h','wind_sum_12h','wind_sum_24h',
    'any_storm_sum_6h','any_storm_sum_12h','any_storm_sum_24h',
    'hour_sin','hour_cos','month_sin','month_cos',
    'is_weekend','quarter',
    'lag_1h_peak','lag_24h_peak','lag_168h_peak',
    'lag_24h_label','lag_168h_label',
    'county_fragility','null_count',
    'county_p90','county_mean_baseline'
]

# Train: 2015-2020, Val: 2021, Test: 2022
splits = {
    'train': list(range(2015, 2021)),
    'val':   [2021],
    'test':  [2022]
}

for split, years in splits.items():
    dfs = []
    for yr in years:
        fp = FEATURES / f'feature_store_{yr}.parquet'
        if not fp.exists(): continue
        df = pd.read_parquet(fp, columns=['fips','hour','outage_label',
                                          'peak_customers_out','log_peak_customers',
                                          'year','state'] + FEATURE_COLS)
        # Stratified sample for train to keep memory manageable
        if split == 'train':
            pos = df[df['outage_label'] == 1]
            neg = df[df['outage_label'] == 0].sample(
                min(len(pos) * 4, len(df[df['outage_label']==0])),
                random_state=42
            )
            df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        dfs.append(df)
        print(f"  {split} {yr}: {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(FEATURES / f'{split}.parquet', index=False)
    pos_rate = combined['outage_label'].mean() * 100
    print(f"  → {split}.parquet: {len(combined):,} rows | outage rate: {pos_rate:.2f}%")
    del combined, dfs

print("\n" + "=" * 60)
print("FEATURE STORE BUILD COMPLETE")
print("=" * 60)
print(f"Output: {FEATURES}")
print("Files created:")
for f in sorted(FEATURES.glob('*.parquet')):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:40s} {size_mb:8.1f} MB")