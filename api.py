from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import os

from risk_engine   import get_risk_score, get_all_risk_scores, HIGHWAYS
from alert_engine  import generate_alert, generate_all_alerts
from map_builder   import build_live_risk_map
from hotspot_detector import get_hotspot_summary, load_accident_data, run_dbscan

app = FastAPI(
    title="RoadSense India API",
    description="Real-time road safety intelligence for Indian highways",
    version="1.0.0"
)

# Allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "project": "RoadSense India",
        "status":  "running",
        "endpoints": [
            "/risk/{highway_name}",
            "/risk/all",
            "/alert/{highway_name}",
            "/hotspots",
            "/map",
        ]
    }


# ── Get risk score for one highway ───────────────────────────────────────
@app.get("/risk/{highway_name}")
def risk_one(highway_name: str):
    # Match partial highway name
    matched = [hw for hw in HIGHWAYS
               if highway_name.lower() in hw.lower()]
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"Highway '{highway_name}' not found. "
                   f"Available: {list(HIGHWAYS.keys())}"
        )
    return get_risk_score(matched[0])


# ── Get risk scores for all highways ─────────────────────────────────────
@app.get("/risk/all/scores")
def risk_all():
    df = get_all_risk_scores()
    return df.to_dict(orient="records")


# ── Generate bilingual alert for one highway ─────────────────────────────
@app.get("/alert/{highway_name}")
def alert_one(highway_name: str):
    matched = [hw for hw in HIGHWAYS
               if highway_name.lower() in hw.lower()]
    if not matched:
        raise HTTPException(status_code=404,
                            detail=f"Highway '{highway_name}' not found.")
    return generate_alert(matched[0])


# ── Get all hotspot clusters ──────────────────────────────────────────────
@app.get("/hotspots")
def hotspots():
    path = "data/hotspot_summary.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=503,
                            detail="Run hotspot_detector.py first.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


# ── Serve the live map as HTML ────────────────────────────────────────────
@app.get("/map", response_class=HTMLResponse)
def live_map():
    path = "data/live_risk_map.html"
    if not os.path.exists(path):
        build_live_risk_map()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Refresh map with new live data ────────────────────────────────────────
@app.get("/map/refresh")
def refresh_map():
    build_live_risk_map()
    return {"status": "Map refreshed", "path": "data/live_risk_map.html"}