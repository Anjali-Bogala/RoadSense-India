import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import folium
from streamlit_folium import st_folium
import os, time

from risk_engine      import get_risk_score, get_all_risk_scores, HIGHWAYS
from alert_engine     import generate_alert
from map_builder      import build_live_risk_map
from hotspot_detector import load_accident_data


# Auto-generate data if files don't exist (for cloud deployment)
def ensure_data_exists():
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists("data/iot_sim_data.csv"):
        from iot_simulator import generate_batch
        generate_batch(100)

    if not os.path.exists("data/merged_data.csv"):
        from merge_data import build_merged_dataset
        build_merged_dataset()

    if not os.path.exists("models/risk_model.pkl"):
        from risk_engine import train_risk_model
        train_risk_model()

    if not os.path.exists("data/clustered_accidents.csv"):
        from hotspot_detector import (load_accident_data,
                                       run_dbscan,
                                       get_hotspot_summary)
        df = load_accident_data()
        df = run_dbscan(df)
        get_hotspot_summary(df)

    if not os.path.exists("data/live_risk_map.html"):
        from map_builder import build_live_risk_map
        build_live_risk_map()

ensure_data_exists()

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadSense India",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a1a2e; text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .sub-title {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .risk-high   { background:#ffe0e0; border-left:5px solid #e74c3c;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .risk-medium { background:#fff3cd; border-left:5px solid #f39c12;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .risk-low    { background:#d4edda; border-left:5px solid #27ae60;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .alert-box   { background:#f8f9fa; border:1px solid #dee2e6;
                   border-radius:8px; padding:16px; margin:10px 0; }
    .hindi-text  { font-size:1.05rem; color:#2c3e50;
                   line-height:1.8; margin-top:8px; }
    .metric-card { text-align:center; padding:12px;
                   background:#f8f9fa; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚦 RoadSense India</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Real-Time Road Safety & Accident Hotspot Intelligence System'
    '</div>',
    unsafe_allow_html=True)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/"
             "4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
             width=80)
    st.markdown("### RoadSense India")
    st.markdown("AI-powered road safety platform for Indian highways.")
    st.divider()
    st.markdown("*Monitored Highways*")
    for hw in HIGHWAYS:
        st.markdown(f"- {hw}")
    st.divider()
    st.markdown("*Data Sources*")
    st.markdown("- data.gov.in (Govt of India)")
    st.markdown("- Open-Meteo Weather API")
    st.markdown("- IoT Traffic Simulation")
    st.markdown("- Llama 3.2 (Ollama)")
    st.divider()
    st.caption("Built with Python · Streamlit · Folium · DBSCAN · LLM")

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 EDA & Insights",
    "🗺️ Live Hotspot Map",
    "🤖 Route Risk Checker"
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & INSIGHTS
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Accident Data Analysis — India")

    # Load data
    acc_path = "data/accidents_clean.csv"
    if not os.path.exists(acc_path):
        st.error("accidents_clean.csv not found. Run eda_analysis.ipynb first.")
        st.stop()

    df = pd.read_csv(acc_path)

    # ── Top metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        state_col = [c for c in df.columns if "state" in c.lower()][0]
        st.metric("States Covered", df[state_col].nunique())
    with col3:
        hotspot_path = "data/hotspot_summary.csv"
        if os.path.exists(hotspot_path):
            hs = pd.read_csv(hotspot_path)
            st.metric("Hotspot Clusters", len(hs))
        else:
            st.metric("Hotspot Clusters", "Run detector")
    with col4:
        st.metric("Highways Monitored", len(HIGHWAYS))

    st.divider()

    # ── Charts row 1 ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Top 10 Accident-Prone States")
        state_counts = (df[state_col]
                        .value_counts()
                        .head(10)
                        .reset_index())
        state_counts.columns = ["State", "Accidents"]
        fig1 = px.bar(
            state_counts, x="Accidents", y="State",
            orientation="h", color="Accidents",
            color_continuous_scale="Reds",
            height=350
        )
        fig1.update_layout(showlegend=False,
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.markdown("#### Accident Distribution by State")
        fig2 = px.pie(
            state_counts, values="Accidents", names="State",
            color_discrete_sequence=px.colors.sequential.RdBu,
            height=350
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Charts row 2 — IoT data ──
    st.markdown("#### Live IoT Sensor Data — Highway Traffic")
    iot_path = "data/merged_data.csv"
    if os.path.exists(iot_path):
        iot_df = pd.read_csv(iot_path)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("*Speed vs Traffic Density*")
            fig3 = px.scatter(
                iot_df, x="speed_kmh", y="density_veh",
                color="incident_flag",
                hover_data=["highway", "condition"],
                color_continuous_scale=["green", "red"],
                height=300,
                labels={"speed_kmh": "Speed (km/h)",
                        "density_veh": "Vehicles/km",
                        "incident_flag": "Incident"}
            )
            fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.markdown("*Weather Conditions Across Highways*")
            weather_counts = (iot_df["condition"]
                              .value_counts()
                              .reset_index())
            weather_counts.columns = ["Condition", "Count"]
            fig4 = px.bar(
                weather_counts, x="Condition", y="Count",
                color="Condition", height=300,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig4.update_layout(showlegend=False,
                               margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Raw data toggle ──
    with st.expander("View Raw Accident Data"):
        st.dataframe(df.head(100), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE HOTSPOT MAP
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Accident Hotspot Map — India")

    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        refresh = st.button("🔄 Refresh Live Data", type="primary")
    with col_info:
        st.caption("Map shows historical accident hotspots + "
                   "live highway risk scores updated from IoT + weather feeds.")

    if refresh:
        with st.spinner("Fetching live data and rebuilding map..."):
            build_live_risk_map()
        st.success("Map refreshed with latest data!")

    st.divider()

    # ── Live risk score cards ──
    st.markdown("#### Live Risk Scores — All Highways")

    risk_data = get_all_risk_scores()
    cols = st.columns(len(HIGHWAYS))

    for i, row in risk_data.iterrows():
        with cols[i]:
            score = row["risk_score"]
            level = row["risk_level"]
            color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[level]
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div style='font-size:1.6rem'>{color}</div>
                    <div style='font-size:0.7rem;color:#888;margin:2px 0'>
                        {row['highway'].split()[0]}
                        {row['highway'].split()[1]}
                    </div>
                    <div style='font-size:1.4rem;font-weight:700'>
                        {score}
                    </div>
                    <div style='font-size:0.75rem;color:#666'>{level}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # ── Folium map ──
    st.markdown("#### Interactive Hotspot Map")
    st.caption("Click any marker for details. "
               "Toggle layers using the control panel on the map.")

    map_path = "data/live_risk_map.html"
    if not os.path.exists(map_path):
        with st.spinner("Building map for first time..."):
            build_live_risk_map()

    # Build and display folium map inline
    from risk_engine import get_risk_score as grs
    from hotspot_detector import load_accident_data, run_dbscan

    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    # Hotspot clusters
    if os.path.exists("data/hotspot_summary.csv"):
        hs_df = pd.read_csv("data/hotspot_summary.csv")
        RISK_COLORS = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        for _, row in hs_df.iterrows():
            color  = RISK_COLORS.get(row["risk_level"], "gray")
            radius = max(8, min(25, int(row["accident_count"]) * 2))
            folium.CircleMarker(
                location=[row["center_lat"], row["center_lon"]],
                radius=radius, color=color,
                fill=True, fill_color=color, fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>Cluster #{int(row['cluster'])}</b><br>"
                    f"Risk: <b style='color:{color}'>"
                    f"{row['risk_level']}</b><br>"
                    f"Accidents: {int(row['accident_count'])}",
                    max_width=200),
                tooltip=(f"{row['risk_level']} — "
                         f"{int(row['accident_count'])} accidents")
            ).add_to(india_map)

    # Live highway markers
    for hw, (lat, lon) in HIGHWAYS.items():
        r     = grs(hw)
        color = r["risk_color"]
        score = r["risk_score"]
        w     = r["weather"]
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=color, icon="car", prefix="fa"),
            popup=folium.Popup(
                f"<b>{hw}</b><br>"
                f"<b style='color:{color}'>Risk: {score}/100 "
                f"({r['risk_level']})</b><br>"
                f"Speed: {r['speed_kmh']} km/h<br>"
                f"Density: {r['density_veh']} vehicles<br>"
                f"Weather: {w['condition']}<br>"
                f"Rain: {w['rain_mm']} mm",
                max_width=230),
            tooltip=f"{hw} — {score}/100"
        ).add_to(india_map)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;">
        <b>RoadSense India</b><br>
        <span style='color:red'>●</span> HIGH &nbsp;
        <span style='color:orange'>●</span> MEDIUM &nbsp;
        <span style='color:green'>●</span> LOW
    </div>"""
    india_map.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(india_map)

    st_folium(india_map, width=None, height=520,
              use_container_width=True)
    


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — ROUTE RISK CHECKER
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Route Risk Checker — AI Safety Advisory")
    st.caption("Select a highway to get a live risk score + "
               "bilingual AI-generated safety alert.")

    st.divider()

    # ── Highway selector ──
    selected_hw = st.selectbox(
        "Select a highway to analyse:",
        options=list(HIGHWAYS.keys()),
        index=0
    )

    check_btn = st.button("🔍 Analyse Route Risk", type="primary")

    if check_btn:
        with st.spinner(f"Analysing {selected_hw}..."):
            risk = get_risk_score(selected_hw)

        st.divider()

        # ── Risk score display ──
        score = risk["risk_score"]
        level = risk["risk_level"]
        color = risk["risk_color"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Risk Score", f"{score}/100")
        with col2:
            st.metric("Risk Level", level)
        with col3:
            st.metric("Avg Speed", f"{risk['speed_kmh']} km/h")
        with col4:
            st.metric("Traffic Density", f"{risk['density_veh']} veh/km")

        # Coloured risk banner
        css_class = (f"risk-{level.lower()}")
        level_emoji = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[level]
        st.markdown(
            f"<div class='{css_class}'>"
            f"<b>{level_emoji} {level} RISK — {selected_hw}</b><br>"
            f"Risk Score: {score}/100 | "
            f"Weather: {risk['weather']['condition']} | "
            f"Rain: {risk['weather']['rain_mm']} mm | "
            f"Visibility: {risk['weather']['visibility_m']} m"
            f"</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # ── Weather details ──
        st.markdown("#### Current Weather Conditions")
        w = risk["weather"]
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            st.metric("Condition",   w["condition"])
        with wc2:
            st.metric("Rainfall",    f"{w['rain_mm']} mm")
        with wc3:
            st.metric("Visibility",  f"{w['visibility_m']} m")
        with wc4:
            st.metric("Wind Speed",  f"{w['wind_kmh']} km/h")

        st.divider()

        # ── LLM Alert ──
        st.markdown("#### AI Safety Alert")

        with st.spinner("Generating bilingual safety advisory via Llama 3.2..."):
            alert = generate_alert(selected_hw)

        # English alert
        st.markdown("*English Advisory*")
        st.markdown(
            f"<div class='alert-box'>{alert['english_alert']}</div>",
            unsafe_allow_html=True
        )

        # Hindi alert
        st.markdown("*हिंदी सलाह (Hindi Advisory)*")
        st.markdown(
            f"<div class='alert-box hindi-text'>"
            f"{alert['hindi_alert']}</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # ── All highways comparison ──
        st.markdown("#### All Highways Risk Comparison")
        with st.spinner("Loading all highway scores..."):
            all_risks = get_all_risk_scores()

        color_map = {"HIGH": "#e74c3c",
                     "MEDIUM": "#f39c12",
                     "LOW": "#27ae60"}
        all_risks["bar_color"] = all_risks["risk_level"].map(color_map)

        fig = px.bar(
            all_risks,
            x="highway", y="risk_score",
            color="risk_level",
            color_discrete_map=color_map,
            labels={"highway": "Highway",
                    "risk_score": "Risk Score (0–100)"},
            height=350,
            text="risk_score"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=-20,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)