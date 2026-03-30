import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import json

HIGHWAYS = {
    "NH-48 Jaipur-Delhi":    (26.9124, 75.7873),
    "NH-44 Delhi-Srinagar":  (28.6139, 77.2090),
    "NH-8 Mumbai-Pune":      (19.0760, 72.8777),
    "NH-66 Bangalore-Kochi": (12.9716, 77.5946),
    "NH-16 Chennai-Kolkata": (13.0827, 80.2707),
}

RISK_COLORS = {
    "HIGH":   "red",
    "MEDIUM": "orange",
    "LOW":    "green"
}

def build_hotspot_map() -> folium.Map:
    print("Building hotspot map...")

    acc_df     = pd.read_csv("data/clustered_accidents.csv")
    summary_df = pd.read_csv("data/hotspot_summary.csv")

    # Base map centered on India
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB positron"   # clean light map style
    )

    # --- Layer 1: Accident heatmap ---
    heat_data = acc_df[["latitude", "longitude"]].dropna().values.tolist()
    HeatMap(
        heat_data,
        name="Accident Heatmap",
        min_opacity=0.3,
        radius=15,
        blur=10,
        gradient={"0.4": "blue", "0.6": "lime",
                  "0.8": "orange", "1.0": "red"}
    ).add_to(india_map)

    # --- Layer 2: Hotspot cluster markers ---
    hotspot_layer = folium.FeatureGroup(name="Hotspot Clusters")

    for _, row in summary_df.iterrows():
        color  = RISK_COLORS.get(row["risk_level"], "gray")
        radius = max(8, min(25, row["accident_count"] * 2))

        folium.CircleMarker(
            location=[row["center_lat"], row["center_lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"""
                <b>Hotspot Cluster #{int(row['cluster'])}</b><br>
                Risk Level : <b style='color:{color}'>{row['risk_level']}</b><br>
                Accidents  : {int(row['accident_count'])}<br>
                Location   : {row['center_lat']:.3f}, {row['center_lon']:.3f}
                """,
                max_width=220
            ),
            tooltip=f"{row['risk_level']} risk — {int(row['accident_count'])} accidents"
        ).add_to(hotspot_layer)

    hotspot_layer.add_to(india_map)

    # --- Layer 3: Highway markers ---
    highway_layer = folium.FeatureGroup(name="Monitored Highways")

    for hw, (lat, lon) in HIGHWAYS.items():
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<b>{hw}</b><br>Live risk score loads in app",
                max_width=200
            ),
            tooltip=hw,
            icon=folium.Icon(color="purple", icon="road",
                             prefix="fa")
        ).add_to(highway_layer)

    highway_layer.add_to(india_map)

    # --- Legend ---
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                border:1px solid #ccc; font-size:13px; font-family:sans-serif">
        <b>Risk Level</b><br>
        <span style="color:red">&#9679;</span> HIGH &nbsp;
        <span style="color:orange">&#9679;</span> MEDIUM &nbsp;
        <span style="color:green">&#9679;</span> LOW<br>
        <span style="color:purple">&#9873;</span> Monitored Highway
    </div>
    """
    india_map.get_root().html.add_child(folium.Element(legend_html))

    # Layer control toggle
    folium.LayerControl(collapsed=False).add_to(india_map)

    india_map.save("data/hotspot_map.html")
    print("Map saved → data/hotspot_map.html")
    print("Open this file in your browser to see it!")
    return india_map


def build_live_risk_map() -> folium.Map:
    """
    Full map: hotspot heatmap + cluster circles +
    live risk score highway markers.
    """
    from risk_engine import get_risk_score

    print("Building live risk map...")

    acc_df     = pd.read_csv("data/clustered_accidents.csv")
    summary_df = pd.read_csv("data/hotspot_summary.csv")

    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    # Heatmap layer
    heat_data = acc_df[["latitude","longitude"]].dropna().values.tolist()
    HeatMap(heat_data, name="Accident Heatmap",
            radius=15, blur=10, min_opacity=0.3,
            gradient={"0.4":"blue","0.65":"lime",
                      "0.8":"orange","1.0":"red"}
    ).add_to(india_map)

    # Hotspot cluster circles
    hotspot_layer = folium.FeatureGroup(name="Hotspot Clusters")
    for _, row in summary_df.iterrows():
        color  = RISK_COLORS.get(row["risk_level"], "gray")
        radius = max(8, min(25, row["accident_count"] * 2))
        folium.CircleMarker(
            location=[row["center_lat"], row["center_lon"]],
            radius=radius, color=color,
            fill=True, fill_color=color, fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>Cluster #{int(row['cluster'])}</b><br>"
                f"Risk: <b style='color:{color}'>{row['risk_level']}</b><br>"
                f"Accidents: {int(row['accident_count'])}",
                max_width=200),
            tooltip=f"{row['risk_level']} — {int(row['accident_count'])} accidents"
        ).add_to(hotspot_layer)
    hotspot_layer.add_to(india_map)

    # Live highway risk markers
    live_layer = folium.FeatureGroup(name="Live Highway Risk")
    for hw, (lat, lon) in HIGHWAYS.items():
        r     = get_risk_score(hw)
        color = r["risk_color"]
        score = r["risk_score"]
        w     = r["weather"]
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=color, icon="car", prefix="fa"),
            popup=folium.Popup(
                f"""
                <b>{hw}</b><br>
                <b style='color:{color}'>Risk: {score}/100
                  ({r['risk_level']})</b><br>
                Speed    : {r['speed_kmh']} km/h<br>
                Density  : {r['density_veh']} vehicles<br>
                Weather  : {w['condition']}<br>
                Rain     : {w['rain_mm']} mm<br>
                Visibility: {w['visibility_m']} m
                """,
                max_width=240),
            tooltip=f"{hw} — Risk {score}/100"
        ).add_to(live_layer)
    live_layer.add_to(india_map)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;font-family:sans-serif">
        <b>RoadSense India</b><br>
        <span style='color:red'>&#9679;</span> HIGH risk &nbsp;
        <span style='color:orange'>&#9679;</span> MEDIUM &nbsp;
        <span style='color:green'>&#9679;</span> LOW<br>
        <small>Click any marker for live details</small>
    </div>"""
    india_map.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(india_map)

    india_map.save("data/live_risk_map.html")
    print("Live map saved → data/live_risk_map.html")
    return india_map


if __name__ == "__main__":
    build_hotspot_map()
    build_live_risk_map()
    

    
