"""
GridGuard AI - Step 2: Data Download
Downloads all required data for the Long Island grid alert system.
Run from project root: python gridguard_download.py
"""
import requests
import json
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd

BASE = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai")

# Create directories
for d in ["data/raw/grid", "data/raw/facilities", "data/raw/storms", "data/graph"]:
    (BASE / d).mkdir(parents=True, exist_ok=True)

GRID_DIR = BASE / "data/raw/grid"
FAC_DIR  = BASE / "data/raw/facilities"
STORM_DIR= BASE / "data/raw/storms"

print("=" * 60)
print("GridGuard AI — Data Download")
print("=" * 60)

# OSM Power Grid Data (Long Island)
print("\n[1/5] Downloading Long Island power grid from OpenStreetMap...")
print("      This may take 3-5 minutes...")

import osmnx as ox
ox.settings.timeout = 180

LONG_ISLAND_BBOX = (40.544, -73.994, 41.027, -71.856)

try:
    print("      Fetching substations...")
    substations = ox.features_from_bbox(
        bbox=LONG_ISLAND_BBOX,
        tags={"power": "substation"}
    )
    substations.to_file(GRID_DIR / "osm_substations.geojson", driver="GeoJSON")
    print(f"      Substations: {len(substations)} found")
except Exception as e:
    print(f"      OSM substation download failed: {e}")
    print("      Creating synthetic substations as fallback...")
    # Synthetic fallback based on known LIPA substations
    synthetic_subs = pd.DataFrame({
        "name": [
            "Mineola Substation", "Hicksville Substation", "Babylon Substation",
            "Riverhead Substation", "Hauppauge Substation", "Bethpage Substation",
            "Garden City Substation", "Hempstead Substation", "Brentwood Substation",
            "Farmingdale Substation", "Ronkonkoma Substation", "Patchogue Substation",
            "Northport Substation", "Port Jefferson Substation", "Lindenhurst Substation",
            "Great Neck Substation", "Freeport Substation", "Massapequa Substation",
            "Coram Substation", "Medford Substation"
        ],
        "voltage": [
            "138", "115", "138", "115", "138", "115",
            "115", "138", "115", "138", "115", "115",
            "138", "115", "115", "115", "115", "115",
            "115", "115"
        ],
        "lat": [
            40.7454, 40.7683, 40.7025, 40.9148, 40.8229, 40.7429,
            40.7335, 40.7062, 40.7812, 40.7315, 40.8217, 40.7650,
            40.9026, 40.9454, 40.6906, 40.7890, 40.6443, 40.6815,
            40.8642, 40.8168
        ],
        "lon": [
            -73.6399, -73.5218, -73.3196, -72.6626, -73.2079, -73.4818,
            -73.6346, -73.6188, -73.2453, -73.4393, -73.1237, -73.0149,
            -73.3429, -73.0493, -73.3729, -73.7271, -73.5832, -73.4588,
            -72.9688, -72.9773
        ],
        "subtype": [
            "transmission", "transmission", "transmission", "transmission",
            "transmission", "distribution", "distribution", "transmission",
            "distribution", "distribution", "distribution", "distribution",
            "transmission", "distribution", "distribution", "distribution",
            "distribution", "distribution", "distribution", "distribution"
        ]
    })
    synthetic_subs.to_csv(GRID_DIR / "synthetic_substations.csv", index=False)
    print(f"      Created {len(synthetic_subs)} synthetic substations")

try:
    time.sleep(2)
    print("      Fetching power lines...")
    power_lines = ox.features_from_bbox(
        bbox=LONG_ISLAND_BBOX,
        tags={"power": ["line", "cable", "minor_line"]}
    )
    power_lines.to_file(GRID_DIR / "osm_power_lines.geojson", driver="GeoJSON")
    print(f"      Power lines: {len(power_lines)} segments found")
except Exception as e:
    print(f"      Power lines download failed (will use proximity edges): {e}")

print("      Grid data download complete.")

# HIFLD Hospitals (Long Island)
print("\n[2/5] Downloading hospitals (HIFLD)...")

HIFLD_HOSPITALS = (
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services"
    "/Hospitals_1/FeatureServer/0/query"
)
params = {
    "where": "STATE='NY' AND COUNTY IN ('NASSAU','SUFFOLK')",
    "outFields": "NAME,ADDRESS,CITY,STATE,ZIP,BEDS,LATITUDE,LONGITUDE,TRAUMA,HELIPAD,OWNER",
    "f": "json",
    "resultRecordCount": 500
}
try:
    resp = requests.get(HIFLD_HOSPITALS, params=params, timeout=30)
    data = resp.json()
    features = data.get("features", [])
    rows = []
    for f in features:
        a = f.get("attributes", {})
        rows.append({
            "name":      a.get("NAME", "Unknown"),
            "address":   a.get("ADDRESS", ""),
            "city":      a.get("CITY", ""),
            "beds":      a.get("BEDS", 0),
            "trauma":    a.get("TRAUMA", "NOT AVAILABLE"),
            "helipad":   a.get("HELIPAD", "N"),
            "lat":       a.get("LATITUDE"),
            "lon":       a.get("LONGITUDE"),
        })
    hosp_df = pd.DataFrame(rows).dropna(subset=["lat","lon"])
    hosp_df.to_csv(FAC_DIR / "hospitals.csv", index=False)
    print(f"      Hospitals saved: {len(hosp_df)}")
except Exception as e:
    print(f"      HIFLD failed ({e}), using curated Long Island hospital list...")
    hospitals_fallback = pd.DataFrame({
        "name": [
            "North Shore University Hospital", "Long Island Jewish Medical Center",
            "Stony Brook University Hospital", "South Shore University Hospital",
            "Good Samaritan Hospital Medical Center", "St. Francis Hospital",
            "Winthrop University Hospital", "NYU Langone Hospital Long Island",
            "Huntington Hospital", "Southside Hospital",
            "Eastern Long Island Hospital", "Peconic Bay Medical Center",
            "St. Catherine of Siena Medical Center", "John T. Mather Memorial Hospital",
            "Glen Cove Hospital"
        ],
        "lat": [
            40.7731, 40.7531, 40.9121, 40.6655, 40.7239, 40.7856,
            40.7454, 40.7193, 40.8812, 40.7325, 41.0076, 40.9192,
            40.8692, 40.9463, 40.8623
        ],
        "lon": [
            -73.5553, -73.7057, -73.1137, -73.5205, -73.1496, -73.6354,
            -73.6388, -73.5965, -73.4256, -73.1538, -72.3044, -72.6543,
            -73.1149, -73.0482, -73.6254
        ],
        "beds": [800, 583, 603, 300, 437, 306, 591, 290, 408, 350, 90, 160, 250, 248, 281],
        "trauma": [
            "LEVEL I","LEVEL II","LEVEL I","NOT AVAILABLE","LEVEL II","LEVEL II",
            "LEVEL II","NOT AVAILABLE","LEVEL III","LEVEL II","NOT AVAILABLE",
            "NOT AVAILABLE","NOT AVAILABLE","NOT AVAILABLE","NOT AVAILABLE"
        ],
        "helipad": ["Y","Y","Y","N","N","Y","Y","N","Y","N","N","N","N","Y","N"]
    })
    hospitals_fallback.to_csv(FAC_DIR / "hospitals.csv", index=False)
    print(f"      Created {len(hospitals_fallback)} hospital records from curated list")

#3. HIFLD Fire Stations
print("\n[3/5] Downloading fire stations (HIFLD)...")

HIFLD_FIRE = (
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services"
    "/Fire_Stations/FeatureServer/0/query"
)
params["where"] = "STATE='NY' AND COUNTY IN ('NASSAU','SUFFOLK')"
params["outFields"] = "NAME,ADDRESS,CITY,COUNTY,LATITUDE,LONGITUDE"
try:
    resp = requests.get(HIFLD_FIRE, params=params, timeout=30)
    data = resp.json()
    rows = []
    for f in data.get("features", []):
        a = f.get("attributes", {})
        rows.append({
            "name":    a.get("NAME","Unknown"),
            "city":    a.get("CITY",""),
            "county":  a.get("COUNTY",""),
            "lat":     a.get("LATITUDE"),
            "lon":     a.get("LONGITUDE"),
        })
    fire_df = pd.DataFrame(rows).dropna(subset=["lat","lon"])
    fire_df.to_csv(FAC_DIR / "fire_stations.csv", index=False)
    print(f"      Fire stations saved: {len(fire_df)}")
except Exception as e:
    print(f"      HIFLD fire stations failed ({e}), using sample data...")
    fire_fallback = pd.DataFrame({
        "name": [
            "Mineola FD", "Garden City FD", "Hempstead FD", "Valley Stream FD",
            "Babylon FD", "Bay Shore FD", "Brentwood FD", "Commack FD",
            "Hauppauge FD", "Huntington FD", "Islip FD", "Lindenhurst FD",
            "Massapequa FD", "Patchogue FD", "Riverhead FD"
        ],
        "lat": [40.7454,40.7268,40.7062,40.6626,40.7025,40.7284,40.7812,
                40.8434,40.8229,40.8695,40.7290,40.6906,40.6815,40.7650,40.9148],
        "lon": [-73.6399,-73.6346,-73.6188,-73.7082,-73.3196,-73.2480,-73.2453,
                -73.3020,-73.2079,-73.4256,-73.2121,-73.3729,-73.4588,-73.0149,-72.6626],
        "county": ["Nassau","Nassau","Nassau","Nassau","Suffolk","Suffolk","Suffolk",
                   "Suffolk","Suffolk","Suffolk","Suffolk","Suffolk","Nassau","Suffolk","Suffolk"]
    })
    fire_fallback.to_csv(FAC_DIR / "fire_stations.csv", index=False)
    print(f"      Created {len(fire_fallback)} fire station records")

#HIFLD Schools (sample)
print("\n[4/5] Downloading school data...")
params["where"] = "STATE='NY' AND COUNTY IN ('NASSAU','SUFFOLK')"
params["outFields"] = "NAME,ADDRESS,CITY,COUNTY,ENROLLMENT,LATITUDE,LONGITUDE"

HIFLD_SCHOOLS = (
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services"
    "/Public_Schools/FeatureServer/0/query"
)
try:
    resp = requests.get(HIFLD_SCHOOLS, params=params, timeout=30)
    data = resp.json()
    rows = []
    for f in data.get("features", []):
        a = f.get("attributes", {})
        rows.append({
            "name":       a.get("NAME","Unknown"),
            "city":       a.get("CITY",""),
            "county":     a.get("COUNTY",""),
            "enrollment": a.get("ENROLLMENT",0),
            "lat":        a.get("LATITUDE"),
            "lon":        a.get("LONGITUDE"),
        })
    school_df = pd.DataFrame(rows).dropna(subset=["lat","lon"])
    # Sample 80 schools for performance
    if len(school_df) > 80:
        school_df = school_df.sample(80, random_state=42)
    school_df.to_csv(FAC_DIR / "schools.csv", index=False)
    print(f"      Schools saved: {len(school_df)}")
except Exception as e:
    print(f"      Schools download failed ({e}), skipping (optional)")
    pd.DataFrame(columns=["name","lat","lon","county","enrollment"]
                ).to_csv(FAC_DIR / "schools.csv", index=False)

#Historical storm polygons
print("\n[5/5] Creating historical storm polygons...")

from shapely.geometry import mapping
from shapely.geometry import Point
import pyproj
from shapely.ops import transform as shapely_transform

def make_storm_circle(center_lat, center_lon, radius_km, name):
    wgs84 = pyproj.CRS("EPSG:4326")
    utm18n = pyproj.CRS("EPSG:32618")
    fwd = pyproj.Transformer.from_crs(wgs84, utm18n, always_xy=True).transform
    inv = pyproj.Transformer.from_crs(utm18n, wgs84, always_xy=True).transform
    pt_utm = shapely_transform(fwd, Point(center_lon, center_lat))
    buf_utm = pt_utm.buffer(radius_km * 1000)
    buf_wgs = shapely_transform(inv, buf_utm)
    return {
        "type": "Feature",
        "properties": {"name": name, "storm": name},
        "geometry": mapping(buf_wgs)
    }

storms = {
    # Sandy: landfall NJ Oct 29 2012, Long Island severely affected
    "sandy_2012": make_storm_circle(40.4, -73.9, 220, "Hurricane Sandy 2012"),
    # Isaias: Long Island August 4, 2020
    "isaias_2020": make_storm_circle(40.7, -73.5, 120, "Tropical Storm Isaias 2020"),
    # Henri: August 2021
    "henri_2021": make_storm_circle(40.9, -72.8, 100, "Tropical Storm Henri 2021"),
}

for name, feature in storms.items():
    with open(STORM_DIR / f"{name}.geojson", "w") as f:
        json.dump({"type":"FeatureCollection","features":[feature]}, f)
    print(f"      Saved storm polygon: {name}")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print(f"\nFiles created in: {BASE / 'data/raw'}")
for folder in ["grid","facilities","storms"]:
    files = list((BASE / "data/raw" / folder).glob("*"))
    print(f"  {folder}/: {[f.name for f in files]}")

print("\nNext step: python gridguard_build_graph.py")
