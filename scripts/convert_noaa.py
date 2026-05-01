# convert_noaa.py
from pathlib import Path
import pandas as pd
import re

raw_dir = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai\data\raw\noaa")
out_dir = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai\data\processed\noaa")
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(raw_dir.glob("StormEvents_details-ftp_v1.0_d*.csv"))
print(f"Files found: {[f.name for f in files]}")

for csv_file in files:
    # Correctly extract year using regex: matches _d followed by 4 digits
    year = re.search(r'_d(\d{4})_', csv_file.name).group(1)
    print(f"\nConverting NOAA details {year}...")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"  Shape: {df.shape}")
    out_path = out_dir / f"noaa_details_{year}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")

print("\nAll done.")