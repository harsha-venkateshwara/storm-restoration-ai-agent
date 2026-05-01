"""
GridGuard AI - Step 4: Alert Engine
K-hop traversal to find critical facilities on predicted outage paths.
Run from project root: python gridguard_alert_engine.py
Saves alerts to: data/graph/alerts_output.json
"""
import pickle
import json
import numpy as np
import networkx as nx
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, shape

BASE      = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai")
GRAPH_DIR = BASE / "data/graph"
STORM_DIR = BASE / "data/raw/storms"

print("=" * 60)
print("GridGuard AI — Alert Engine")
print("=" * 60)

# Load graph
with open(GRAPH_DIR / "long_island_grid.pkl", "rb") as f:
    G = pickle.load(f)
print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Storm scenario multipliers
STORM_MULTIPLIERS = {
    "No Active Storm":     1.00,
    "Thunderstorm":        1.40,
    "Winter Storm":        1.50,
    "Flash Flood":         1.30,
    "High Wind Event":     1.60,
    "Hurricane Cat 1":     2.00,
    "Hurricane Cat 3":     3.50,
    "Hurricane Cat 5":     5.50,
}

# Recommended actions
def get_action(facility_type, risk, hops):
    if facility_type == "hospital":
        if risk > 0.85:
            return "IMMEDIATE ACTION: Activate emergency generators now. Alert backup power team. Notify generator fuel supplier. Contact backup facility for patient diversion plan."
        elif risk > 0.65:
            return "PREPARE: Test emergency generators within 2 hours. Confirm fuel reserves sufficient for 72h. Alert on-call engineering staff. Brief department heads."
        else:
            return "MONITOR: Review generator maintenance logs. Confirm emergency contacts are current. Brief charge nurses on storm protocol."
    elif facility_type == "fire_station":
        if risk > 0.85:
            return "IMMEDIATE ACTION: Deploy mobile generator. Activate backup communications. Pre-position apparatus for storm response. Fuel all vehicles."
        else:
            return "PREPARE: Confirm backup power status. Brief crew on storm protocols. Stage equipment for rapid deployment."
    elif facility_type == "school":
        if risk > 0.75:
            return "CONSIDER CLOSURE: Alert district emergency coordinator. Notify parents of potential early dismissal. Confirm shelter-in-place procedures."
        else:
            return "MONITOR: Review shelter-in-place procedures. Confirm emergency contact lists are updated."
    return "Monitor situation and follow standard emergency protocols."

# Core alert engine
def find_at_risk_substations(G, storm_polygon, storm_type="Hurricane Cat 3"):
    multiplier  = STORM_MULTIPLIERS.get(storm_type, 2.0)
    at_risk     = []
    inside_count= 0

    for node_id, data in G.nodes(data=True):
        if data["node_type"] != "substation":
            continue
        pt = Point(data["lon"], data["lat"])
        in_storm = storm_polygon.contains(pt)
        if in_storm:
            inside_count += 1

        base_risk = data.get("base_risk", 0.45)
        if in_storm:
            adjusted = min(0.999, base_risk * multiplier)
        else:
            # Partial risk for substations near storm edge
            dist_deg = storm_polygon.boundary.distance(pt)
            dist_km  = dist_deg * 111
            if dist_km < 30:
                adjusted = min(0.999, base_risk * multiplier * (1 - dist_km/60))
            else:
                adjusted = base_risk

        if adjusted > 0.40:
            at_risk.append({
                "node_id":       node_id,
                "name":          data.get("name", node_id),
                "lat":           data["lat"],
                "lon":           data["lon"],
                "voltage":       data.get("voltage","115"),
                "subtype":       data.get("subtype","distribution"),
                "base_risk":     round(base_risk, 4),
                "adjusted_risk": round(adjusted, 4),
                "in_storm_zone": in_storm,
            })

    at_risk.sort(key=lambda x: x["adjusted_risk"], reverse=True)
    print(f"Substations in storm zone: {inside_count}")
    print(f"At-risk substations (P>0.40): {len(at_risk)}")
    return at_risk


CRITICAL_TYPES = {
    "hospital":    {"priority":"CRITICAL", "score":10, "max_hops":2},
    "fire_station":{"priority":"HIGH",     "score":7,  "max_hops":3},
    "school":      {"priority":"MEDIUM",   "score":4,  "max_hops":3},
}

def traverse_and_alert(G, at_risk_substations, k_hops=3):
    alerts = {}

    for sub_info in at_risk_substations:
        sub_id   = sub_info["node_id"]
        sub_risk = sub_info["adjusted_risk"]

        try:
            reachable = nx.single_source_shortest_path_length(
                G, sub_id, cutoff=k_hops
            )
        except Exception:
            continue

        for node_id, hop_dist in reachable.items():
            if node_id == sub_id or hop_dist == 0:
                continue
            ndata = G.nodes.get(node_id, {})
            ntype = ndata.get("node_type","")
            if ntype not in CRITICAL_TYPES:
                continue
            cfg = CRITICAL_TYPES[ntype]
            if hop_dist > cfg["max_hops"]:
                continue

            alert_score = (sub_risk * cfg["score"]) / (hop_dist ** 0.5)
            if node_id in alerts and alert_score <= alerts[node_id]["alert_score"]:
                continue

            props = ndata.get("props", {})
            alerts[node_id] = {
                "facility_id":        node_id,
                "facility_name":      ndata.get("name", node_id),
                "facility_type":      ntype,
                "priority":           cfg["priority"],
                "lat":                ndata.get("lat", 0),
                "lon":                ndata.get("lon", 0),
                "feeding_substation": sub_info["name"],
                "feeding_sub_lat":    sub_info["lat"],
                "feeding_sub_lon":    sub_info["lon"],
                "substation_risk":    round(sub_risk, 4),
                "hop_distance":       hop_dist,
                "alert_score":        round(alert_score, 4),
                "in_storm_zone":      sub_info["in_storm_zone"],
                "beds":               props.get("beds", props.get("BEDS", 0)),
                "trauma":             props.get("trauma", props.get("TRAUMA","N/A")),
                "action":             get_action(ntype, sub_risk, hop_dist),
            }

    sorted_alerts = sorted(alerts.values(),
                           key=lambda x: x["alert_score"], reverse=True)
    return sorted_alerts


#Run with Sandy storm polygon
def load_storm(storm_name="sandy_2012"):
    path = STORM_DIR / f"{storm_name}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Storm file not found: {path}")
    with open(path) as f:
        fc = json.load(f)
    geom = fc["features"][0]["geometry"]
    return shape(geom)

if __name__ == "__main__":
    print("\nRunning alert engine with Hurricane Sandy polygon...")
    storm_poly  = load_storm("sandy_2012")
    at_risk     = find_at_risk_substations(G, storm_poly, "Hurricane Cat 3")
    alerts      = traverse_and_alert(G, at_risk, k_hops=3)

    print(f"\nTotal alerts generated: {len(alerts)}")
    critical = [a for a in alerts if a["priority"]=="CRITICAL"]
    high     = [a for a in alerts if a["priority"]=="HIGH"]
    medium   = [a for a in alerts if a["priority"]=="MEDIUM"]
    print(f"  CRITICAL (hospitals): {len(critical)}")
    print(f"  HIGH (fire stations): {len(high)}")
    print(f"  MEDIUM (schools):     {len(medium)}")

    if critical:
        print("\nTop CRITICAL alerts:")
        for a in critical[:5]:
            print(f"  {a['facility_name']}")
            print(f"    Risk: {a['substation_risk']:.3f} | "
                  f"Hops: {a['hop_distance']} | "
                  f"Fed by: {a['feeding_substation']}")
            print(f"    Action: {a['action'][:80]}...")

    # Save results
    output = {
        "storm":          "sandy_2012",
        "storm_type":     "Hurricane Cat 3",
        "at_risk_substations": at_risk,
        "alerts":         alerts,
        "summary": {
            "total_at_risk_substations": len(at_risk),
            "total_alerts":   len(alerts),
            "critical_count": len(critical),
            "high_count":     len(high),
            "medium_count":   len(medium),
        }
    }
    with open(GRAPH_DIR / "alerts_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAlerts saved to: {GRAPH_DIR / 'alerts_output.json'}")
    print("\nNext step: python gridguard_map.py")
