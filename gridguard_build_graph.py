"""
GridGuard AI - Step 3: Build Grid Topology Graph
Builds the NetworkX graph from downloaded data.
Run from project root: python gridguard_build_graph.py
"""
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
import pickle
import json
from pathlib import Path
from shapely.geometry import Point
from scipy.spatial import cKDTree

BASE      = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai")
GRID_DIR  = BASE / "data/raw/grid"
FAC_DIR   = BASE / "data/raw/facilities"
GRAPH_DIR = BASE / "data/graph"
GRAPH_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("GridGuard AI — Building Grid Topology Graph")
print("=" * 60)

#Loadingsubstation data
print("\n[1/5] Loading substation data...")

G = nx.Graph()

# Trying OSM substations first, fall back to synthetic
osm_path = GRID_DIR / "osm_substations.geojson"
syn_path  = GRID_DIR / "synthetic_substations.csv"

if osm_path.exists():
    subs = gpd.read_file(osm_path).to_crs(epsg=4326)
    for idx, row in subs.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        centroid = geom.centroid if geom.geom_type != "Point" else geom
        osm_id   = str(row.get("osmid", idx))
        name     = row.get("name") or f"Substation_{osm_id}"
        voltage  = str(row.get("voltage", "115"))
        node_id  = f"sub_{osm_id}"
        subtype  = "transmission" if any(v in voltage for v in ["138","230","345"]) else "distribution"
        G.add_node(node_id,
            node_type="substation",
            name=name,
            voltage=voltage,
            subtype=subtype,
            lat=centroid.y,
            lon=centroid.x,
        )
    print(f"      OSM substations loaded: {G.number_of_nodes()}")
elif syn_path.exists():
    df = pd.read_csv(syn_path)
    for idx, row in df.iterrows():
        node_id = f"sub_{idx}"
        G.add_node(node_id,
            node_type="substation",
            name=row["name"],
            voltage=str(row.get("voltage","115")),
            subtype=row.get("subtype","distribution"),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
        )
    print(f"      Synthetic substations loaded: {G.number_of_nodes()}")
else:
    raise FileNotFoundError("No substation data found. Run gridguard_download.py first.")

#Step 2: Add edges from OSM power lines
print("\n[2/5] Adding power line edges...")

lines_path = GRID_DIR / "osm_power_lines.geojson"
edges_added = 0

if lines_path.exists():
    lines = gpd.read_file(lines_path).to_crs(epsg=4326)
    sub_nodes  = [(n,d) for n,d in G.nodes(data=True)]
    sub_ids    = [n for n,d in sub_nodes]
    sub_coords = np.array([(d["lat"],d["lon"]) for n,d in sub_nodes])
    tree       = cKDTree(sub_coords)

    for _, line in lines.iterrows():
        geom = line.geometry
        if geom is None:
            continue
        pts = list(geom.coords) if hasattr(geom, "coords") else []
        if len(pts) < 2:
            continue
        # Connect nearest substations to line start and end
        for pt in [pts[0], pts[-1]]:
            query  = np.array([[pt[1], pt[0]]])
            dist, idx = tree.query(query, k=1)
            dist_km = dist[0][0] * 111
            if dist_km < 5:
                nearest = sub_ids[idx[0][0]]
                # Find another substation on the other end
                dist2, idx2 = tree.query(query, k=3)
                for d2, i2 in zip(dist2[0][1:], idx2[0][1:]):
                    d2_km = d2 * 111
                    if d2_km < 30 and i2 != idx[0][0]:
                        second = sub_ids[i2]
                        if not G.has_edge(nearest, second):
                            G.add_edge(nearest, second,
                                edge_type="transmission",
                                distance_km=round(d2_km, 2),
                                voltage=str(line.get("voltage","115"))
                            )
                            edges_added += 1
                        break
    print(f"      Edges from OSM lines: {edges_added}")
else:
    print("      No OSM lines found. Connecting substations by proximity...")
    sub_ids    = [n for n,d in G.nodes(data=True)]
    sub_coords = np.array([(d["lat"],d["lon"]) for n,d in G.nodes(data=True)])
    tree       = cKDTree(sub_coords)
    for i, node_id in enumerate(sub_ids):
        dists, idxs = tree.query([sub_coords[i]], k=4)
        for dist_deg, j in zip(dists[0][1:], idxs[0][1:]):
            dist_km = dist_deg * 111
            if dist_km < 25:
                neighbor = sub_ids[j]
                if not G.has_edge(node_id, neighbor):
                    G.add_edge(node_id, neighbor,
                        edge_type="distribution",
                        distance_km=round(dist_km,2)
                    )
                    edges_added += 1
    print(f"      Proximity edges added: {edges_added}")

# Step 3: Add critical facilities
print("\n[3/5] Adding critical facilities...")

FACILITY_CONFIG = [
    ("hospitals.csv",     "hospital",      "CRITICAL", 10),
    ("fire_stations.csv", "fire_station",  "HIGH",     7),
    ("schools.csv",       "school",        "MEDIUM",   4),
]

sub_ids    = [n for n,d in G.nodes(data=True) if d["node_type"]=="substation"]
sub_coords = np.array([(d["lat"],d["lon"]) for n,d in G.nodes(data=True) if d["node_type"]=="substation"])
tree       = cKDTree(sub_coords)

fac_count = 0
for fname, ftype, priority, score in FACILITY_CONFIG:
    fpath = FAC_DIR / fname
    if not fpath.exists():
        print(f"      Skipping {fname} (not found)")
        continue
    df = pd.read_csv(fpath).dropna(subset=["lat","lon"])
    for idx, row in df.iterrows():
        lat = float(row["lat"]); lon = float(row["lon"])
        node_id = f"{ftype}_{idx}"
        props = row.to_dict()
        G.add_node(node_id,
            node_type=ftype,
            name=str(row.get("name", f"{ftype}_{idx}")),
            priority=priority,
            priority_score=score,
            lat=lat, lon=lon,
            props=props
        )
        # Connect to 2 nearest substations
        query = np.array([[lat, lon]])
        dists, idxs = tree.query(query, k=2)
        for dist_deg, j in zip(dists[0], idxs[0]):
            dist_km = dist_deg * 111
            if dist_km < 20:
                nearest_sub = sub_ids[j]
                G.add_edge(nearest_sub, node_id,
                    edge_type="service",
                    distance_km=round(dist_km,2)
                )
        fac_count += 1
    print(f"      {ftype}: {len(df)} loaded")

print(f"      Total facility nodes: {fac_count}")

# Attaching risk scores from existing model
print("\n[4/5] Attaching outage risk scores from existing model...")

county_priority_path = BASE / "outputs/models/county_priority_2022.parquet"
if county_priority_path.exists():
    import pandas as pd
    county_data = pd.read_parquet(county_priority_path)
    # Nassau FIPS=36059, Suffolk FIPS=36103
    li_counties = county_data[
        county_data["fips"].astype(str).str[:5].isin(["36059","36103"])
    ]
    if len(li_counties) > 0:
        avg_risk = float(li_counties["mean_outage_proba"].mean())
        max_risk = float(li_counties["max_outage_proba"].max())
        print(f"      LI baseline avg risk: {avg_risk:.4f}, max: {max_risk:.4f}")
    else:
        avg_risk = 0.45
        max_risk = 0.75
        print("      Using default risk scores (LI counties not in test set)")
else:
    avg_risk = 0.45
    max_risk = 0.75
    print("      Model output not found, using default risk scores")

# Assign risk to substations based on proximity weighting
storm_centers = {
    "nassau": (40.6943, -73.5944),
    "suffolk": (40.8518, -73.1128),
}
for node_id, data in G.nodes(data=True):
    if data["node_type"] != "substation":
        continue
    lat, lon = data["lat"], data["lon"]
    # Nassau/western LI gets higher risk (closer to NYC/Sandy path)
    if lon > -73.4:  # eastern Suffolk
        risk = avg_risk * 0.85
    elif lon > -73.6:  # central
        risk = avg_risk
    else:  # western Nassau
        risk = avg_risk * 1.15
    G.nodes[node_id]["base_risk"] = round(min(0.99, risk), 4)

#Saving graph
print("\n[5/5] Saving graph...")

with open(GRAPH_DIR / "long_island_grid.pkl", "wb") as f:
    pickle.dump(G, f)

# Summary
from collections import Counter
type_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))
print("\n" + "=" * 60)
print("GRAPH BUILD COMPLETE")
print("=" * 60)
print(f"Total nodes : {G.number_of_nodes():,}")
print(f"Total edges : {G.number_of_edges():,}")
print("\nNode breakdown:")
for t, c in type_counts.most_common():
    print(f"  {t:20s}: {c}")
print(f"\nGraph saved to: {GRAPH_DIR / 'long_island_grid.pkl'}")
print("\nNext step: python gridguard_alert_engine.py")
