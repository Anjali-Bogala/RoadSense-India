import numpy as np
import pandas as pd
import pickle, os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from weather_api import get_weather, HIGHWAYS
from iot_simulator import simulate_reading

# ── Train the model (runs once, saves to disk) ──────────────────────────
def train_risk_model():
    print("Training risk model...")
    np.random.seed(42)
    n = 2000

    speed    = np.random.normal(65, 20, n).clip(10, 120)
    density  = np.random.randint(10, 200, n).astype(float)
    rain     = np.random.choice([0, 0, 0, 1], n).astype(float)
    fog      = np.random.choice([0, 0, 0, 0, 1], n).astype(float)
    incident = np.random.choice([0, 0, 0, 0, 0, 1], n).astype(float)

    # Ground truth risk formula
    risk = (
        (100 - speed) / 100 * 30 +
        density / 200 * 30 +
        rain    * 20 +
        fog     * 15 +
        incident * 5
    ).clip(0, 100)

    X = np.column_stack([speed, density, rain, fog, incident])

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, risk)

    os.makedirs("models", exist_ok=True)
    pickle.dump(model,  open("models/risk_model.pkl",  "wb"))
    pickle.dump(scaler, open("models/risk_scaler.pkl", "wb"))
    print(f"Model trained — R²: {model.score(X_scaled, risk):.4f}")
    print("Saved → models/risk_model.pkl")
    return model, scaler


# ── Load trained model ───────────────────────────────────────────────────
def load_model():
    if not os.path.exists("models/risk_model.pkl"):
        train_risk_model()
    model  = pickle.load(open("models/risk_model.pkl",  "rb"))
    scaler = pickle.load(open("models/risk_scaler.pkl", "rb"))
    return model, scaler


# ── Main function: get risk score for one highway ────────────────────────
def get_risk_score(highway: str) -> dict:
    model, scaler = load_model()

    # Live IoT reading
    iot = simulate_reading(highway)
    speed    = iot["speed_kmh"]
    density  = iot["density_veh"]
    incident = iot["incident_flag"]

    # Live weather
    weather = get_weather(highway)

    # Convert weather into model features
    rain = 1 if weather.get("rain_mm", 0) > 0 else 0

    # Simple fog logic based on visibility
    fog = 1 if weather.get("visibility_m", 10000) < 2000 else 0

    # Predict
    X        = np.array([[speed, density, rain, fog, incident]])
    X_scaled = scaler.transform(X)
    score    = float(model.predict(X_scaled)[0])
    score    = round(np.clip(score, 0, 100), 1)

    # Risk level
    level = ("HIGH"   if score > 65 else
             "MEDIUM" if score > 35 else "LOW")

    # Risk color
    color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[level]

    return {
        "highway":     highway,
        "risk_score":  score,
        "risk_level":  level,
        "risk_color":  color,
        "speed_kmh":   round(speed, 1),
        "density_veh": int(density),
        "incident":    bool(incident),
        "weather":     weather,
    }


# ── Get scores for ALL highways ──────────────────────────────────────────
def get_all_risk_scores() -> pd.DataFrame:
    rows = [get_risk_score(hw) for hw in HIGHWAYS]
    df   = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    train_risk_model()
    print("\nLive risk scores:")
    for hw in HIGHWAYS:
        r = get_risk_score(hw)
        print(f"  {r['highway']:<28} "
              f"Score: {r['risk_score']:5.1f}  "
              f"Level: {r['risk_level']}")