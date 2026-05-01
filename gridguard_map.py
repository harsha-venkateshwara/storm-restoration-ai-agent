"""
GridGuard AI - Step 5: Generate Interactive Map
Creates the Folium route map and saves as HTML.
Run from project root: python gridguard_map.py
Output: outputs/gridguard_map.html
"""
import pickle
import json
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import networkx as nx
from pathlib import Path
from shapely.geometry import shape

BASE      = Path(r"C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai")
GRAPH_DIR = BASE / "data/graph"
STORM_DIR = BASE / "data/raw/storms"
OUT_DIR   = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("GridGuard AI — Building Interactive Map")
print("=" * 60)

# Load data
with open(GRAPH_DIR / "long_island_grid.pkl", "rb") as f:
    G = pickle.load(f)

with open(GRAPH_DIR / "alerts_output.json") as f:
    alert_data = json.load(f)

at_risk  = alert_data["at_risk_substations"]
alerts   = alert_data["alerts"]
at_risk_ids = {s["node_id"] for s in at_risk}

with open(STORM_DIR / "sandy_2012.geojson") as f:
    storm_fc = json.load(f)
storm_geom = shape(storm_fc["features"][0]["geometry"])

print(f"Loaded graph: {G.number_of_nodes()} nodes")
print(f"Loaded alerts: {len(alerts)}")

# Build map
m = folium.Map(
    location=[40.75, -73.2],
    zoom_start=10,
    tiles="CartoDB dark_matter"
)

#Storm polygon
folium.GeoJson(
    storm_fc,
    style_function=lambda x: {
        "fillColor":    "#ef4444",
        "fillOpacity":  0.12,
        "color":        "#ef4444",
        "weight":       2.5,
        "dashArray":    "6 4",
    },
    tooltip="Hurricane Sandy Impact Zone"
).add_to(m)

# Power lines / edges
line_layer = folium.FeatureGroup(name="Power Lines", show=True)
for u, v, edata in G.edges(data=True):
    u_data = G.nodes.get(u, {}); v_data = G.nodes.get(v, {})
    if ("lat" not in u_data or "lat" not in v_data):
        continue
    # Only draw substation-to-substation lines
    if u_data.get("node_type") != "substation" or v_data.get("node_type") != "substation":
        continue
    both_at_risk = (u in at_risk_ids and v in at_risk_ids)
    color  = "#ef4444" if both_at_risk else "#1d4ed8"
    weight = 2.0 if both_at_risk else 1.2
    opacity= 0.85 if both_at_risk else 0.5
    folium.PolyLine(
        locations=[(u_data["lat"], u_data["lon"]),
                   (v_data["lat"], v_data["lon"])],
        color=color, weight=weight, opacity=opacity,
        tooltip=f"{edata.get('edge_type','line')} | {edata.get('distance_km','?')} km"
    ).add_to(line_layer)
line_layer.add_to(m)

# Substations
sub_layer = folium.FeatureGroup(name="Substations", show=True)
for node_id, data in G.nodes(data=True):
    if data.get("node_type") != "substation":
        continue
    at_risk_entry = next((s for s in at_risk if s["node_id"]==node_id), None)
    risk = at_risk_entry["adjusted_risk"] if at_risk_entry else 0
    in_zone = at_risk_entry["in_storm_zone"] if at_risk_entry else False

    if in_zone:
        color = "#ef4444"; radius = 12; fill_op = 0.9
    elif risk > 0.40:
        color = "#f59e0b"; radius = 9; fill_op = 0.8
    else:
        color = "#3b82f6"; radius = 6; fill_op = 0.6

    popup_html = f"""
    <div style='font-family:sans-serif;min-width:180px;'>
      <b style='color:{color};'>{data.get('name','Substation')}</b><br/>
      Voltage: {data.get('voltage','?')} kV<br/>
      Type: {data.get('subtype','?')}<br/>
      Risk Score: <b>{risk:.3f}</b><br/>
      Storm Zone: {'YES' if in_zone else 'NO'}
    </div>"""

    folium.CircleMarker(
        location=[data["lat"], data["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=fill_op,
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"{data.get('name','Sub')} | Risk: {risk:.3f}"
    ).add_to(sub_layer)
sub_layer.add_to(m)

# Alert routing lines
routing_layer = folium.FeatureGroup(name="Alert Routes", show=True)
PRIORITY_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f59e0b",
    "MEDIUM":   "#3b82f6",
}
for alert in alerts:
    color = PRIORITY_COLORS.get(alert["priority"], "#94a3b8")
    # Dashed line from substation to facility
    folium.PolyLine(
        locations=[
            (alert["feeding_sub_lat"], alert["feeding_sub_lon"]),
            (alert["lat"],             alert["lon"])
        ],
        color=color,
        weight=2.0,
        opacity=0.7,
        dash_array="8 5",
        tooltip=f"Route: {alert['feeding_substation']} → {alert['facility_name']}"
    ).add_to(routing_layer)
routing_layer.add_to(m)

# Critical facility alerts
ICONS = {
    "hospital":    ("plus-square", "red"),
    "fire_station":("fire",        "orange"),
    "school":      ("graduation-cap", "blue"),
}
alert_layer = folium.FeatureGroup(name="Critical Facility Alerts", show=True)

for alert in alerts:
    ftype  = alert["facility_type"]
    icon_n, icon_c = ICONS.get(ftype, ("exclamation-triangle", "gray"))
    color  = PRIORITY_COLORS.get(alert["priority"], "#94a3b8")

    beds_info = ""
    if ftype == "hospital" and alert.get("beds"):
        beds_info = f"<br/>Beds: <b>{alert['beds']}</b>"
        if alert.get("trauma") and alert["trauma"] not in ["NOT AVAILABLE","N/A",""]:
            beds_info += f" | Trauma: <b>{alert['trauma']}</b>"

    popup_html = f"""
    <div style='font-family:sans-serif;min-width:240px;max-width:300px;'>
      <div style='background:{color};color:white;padding:6px 10px;
                  border-radius:4px 4px 0 0;font-weight:bold;'>
        {alert['priority']} ALERT
      </div>
      <div style='padding:8px 10px;border:1px solid #e2e8f0;'>
        <b>{alert['facility_name']}</b>{beds_info}<br/>
        <span style='color:#64748b;font-size:11px;'>
          Feeding substation: {alert['feeding_substation']}<br/>
          Substation risk: {alert['substation_risk']:.3f}<br/>
          Distance (hops): {alert['hop_distance']}<br/>
          Alert score: {alert['alert_score']:.3f}
        </span>
        <hr style='margin:6px 0;border-color:#e2e8f0;'/>
        <b style='color:#1d4ed8;'>Recommended Action:</b><br/>
        <span style='font-size:11px;'>{alert['action']}</span>
      </div>
    </div>"""

    folium.Marker(
        location=[alert["lat"], alert["lon"]],
        popup=folium.Popup(popup_html, max_width=320),
        tooltip=f"{alert['priority']}: {alert['facility_name']}",
        icon=folium.Icon(
            color=icon_c,
            icon=icon_n,
            prefix="fa"
        )
    ).add_to(alert_layer)

alert_layer.add_to(m)

# Legend
legend_html = """
<div style='position:fixed;bottom:30px;left:20px;z-index:1000;
            background:rgba(15,23,42,0.92);color:white;
            padding:12px 16px;border-radius:8px;
            border:1px solid #1e293b;font-family:sans-serif;
            font-size:12px;min-width:200px;'>
  <b style='color:#60a5fa;font-size:14px;'>GridGuard AI</b><br/>
  <span style='color:#94a3b8;font-size:10px;'>Long Island Grid Alert System</span>
  <hr style='border-color:#1e293b;margin:8px 0;'/>
  <div><span style='color:#ef4444;'>&#9679;</span> At-Risk Substation</div>
  <div><span style='color:#f59e0b;'>&#9679;</span> Elevated Risk Substation</div>
  <div><span style='color:#3b82f6;'>&#9679;</span> Normal Substation</div>
  <hr style='border-color:#1e293b;margin:8px 0;'/>
  <div><span style='color:#ef4444;'>&#9632;</span> CRITICAL Alert (Hospital)</div>
  <div><span style='color:#f59e0b;'>&#9632;</span> HIGH Alert (Fire Station)</div>
  <div><span style='color:#3b82f6;'>&#9632;</span> MEDIUM Alert (School)</div>
  <hr style='border-color:#1e293b;margin:8px 0;'/>
  <div style='color:#94a3b8;font-size:10px;'>
    Click any marker for details<br/>
    Dashed lines = alert routing paths
  </div>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

# Layer control 
folium.LayerControl(collapsed=False).add_to(m)

# Save map 
map_path = OUT_DIR / "gridguard_map.html"
m.save(str(map_path))

print(f"\nMap saved: {map_path}")
print(f"Open in browser: file:///{str(map_path).replace(chr(92), '/')}")
print("\nSummary:")
print(f"  Storm: Hurricane Sandy 2012")
print(f"  At-risk substations: {len(at_risk)}")
print(f"  Total alerts: {len(alerts)}")
print(f"  Critical (hospitals): {len([a for a in alerts if a['priority']=='CRITICAL'])}")
print(f"  High (fire stations): {len([a for a in alerts if a['priority']=='HIGH'])}")
print(f"  Medium (schools):     {len([a for a in alerts if a['priority']=='MEDIUM'])}")
print("\nNext step: Add the GridGuard tab to your Streamlit app.py")
