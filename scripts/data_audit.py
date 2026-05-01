# scripts/data_audit.py
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai")
eaglei_dir = BASE / "data/processed/eagle_i"
noaa_dir   = BASE / "data/processed/noaa"

# EAGLE-I AUDIT
print("=" * 60)
print("EAGLE-I AUDIT")
print("=" * 60)

dfs = []
for f in sorted(eaglei_dir.glob("eaglei_*.parquet")):
    df = pd.read_parquet(f)
    year = f.stem.split("_")[-1]
    df["run_start_time"] = pd.to_datetime(df["run_start_time"])
    nulls = df.isnull().sum()
    print(f"\n[{year}]")
    print(f"  Rows         : {len(df):,}")
    print(f"  Counties     : {df['fips_code'].nunique()}")
    print(f"  Date range   : {df['run_start_time'].min()} → {df['run_start_time'].max()}")
    print(f"  'sum' nulls  : {nulls['sum']}")
    print(f"  'sum' > 0    : {(df['sum'] > 0).sum():,} ({(df['sum'] > 0).mean()*100:.1f}%)")
    print(f"  'sum' max    : {df['sum'].max():,}")
    dfs.append(df)

eaglei = pd.concat(dfs, ignore_index=True)
print(f"\nTotal EAGLE-I rows : {len(eaglei):,}")
print(f"Total unique FIPS  : {eaglei['fips_code'].nunique()}")

# 2. NOAA AUDIT
print("\n" + "=" * 60)
print("NOAA AUDIT")
print("=" * 60)

noaa_dfs = []
for f in sorted(noaa_dir.glob("noaa_details_*.parquet")):
    df = pd.read_parquet(f)
    year = f.stem.split("_")[-1]
    print(f"\n[{year}]")
    print(f"  Rows           : {len(df):,}")
    print(f"  Event types    : {df['EVENT_TYPE'].nunique()}")
    print(f"  CZ_TYPE counts : {df['CZ_TYPE'].value_counts().to_dict()}")
    print(f"  STATE_FIPS nulls: {df['STATE_FIPS'].isnull().sum()}")
    print(f"  CZ_FIPS nulls  : {df['CZ_FIPS'].isnull().sum()}")
    noaa_dfs.append(df)

noaa = pd.concat(noaa_dfs, ignore_index=True)
print(f"\nTop 15 EVENT_TYPEs across all years:")
print(noaa['EVENT_TYPE'].value_counts().head(15))

print(f"\nTotal NOAA rows : {len(noaa):,}")
print(f"Unique states   : {noaa['STATE'].nunique()}")