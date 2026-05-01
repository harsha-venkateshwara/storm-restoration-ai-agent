from pathlib import Path
import pandas as pd

raw_dir = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai\data\raw\eagle_i\eaglei_outages")
out_dir = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai\data\processed\eagle_i")
out_dir.mkdir(parents=True, exist_ok=True)

# Sanity check
print("Looking in:", raw_dir.resolve())
print("Exists:", raw_dir.exists())
print("Files found:", [f.name for f in raw_dir.glob("eaglei_outages_*.csv")])

for csv_file in sorted(raw_dir.glob("eaglei_outages_*.csv")):
    year = csv_file.stem.split("_")[-1]
    print(f"\nConverting {year}...")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"  Shape: {df.shape} | Columns: {list(df.columns)}")
    out_path = out_dir / f"eaglei_{year}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")

print("\nAll done.")