import requests
import pandas as pd
import time

# Disable SSL warnings (helps avoid SSL errors on Windows)
requests.packages.urllib3.disable_warnings()

HIGHWAYS = {
    "NH-48 Jaipur-Delhi":    (26.9124, 75.7873),
    "NH-44 Delhi-Srinagar":  (28.6139, 77.2090),
    "NH-8 Mumbai-Pune":      (19.0760, 72.8777),
    "NH-66 Bangalore-Kochi": (12.9716, 77.5946),
    "NH-16 Chennai-Kolkata": (13.0827, 80.2707),
}

def get_weather(highway: str) -> dict:
    lat, lon = HIGHWAYS[highway]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current":   ["rain", "visibility", "wind_speed_10m",
                      "weather_code", "temperature_2m"],
        "timezone":  "Asia/Kolkata"
    }

    try:
        # 🔁 Retry mechanism (3 attempts)
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=5, verify=False)
                r.raise_for_status()
                break
            except Exception as e:
                print(f"Retry {attempt+1} for {highway} failed: {e}")
                time.sleep(1)
        else:
            raise Exception("All retries failed")

        c = r.json()["current"]

        rain_mm      = c.get("rain", 0) or 0
        visibility_m = c.get("visibility", 10000) or 10000
        wind_kmh     = c.get("wind_speed_10m", 0) or 0
        temp_c       = c.get("temperature_2m", 25) or 25
        weather_code = c.get("weather_code", 0) or 0

        fog_flag  = 1 if visibility_m < 1000 else 0
        rain_flag = 1 if rain_mm > 0.5 else 0
        condition = get_condition_label(weather_code)

        return {
            "highway":      highway,
            "rain_mm":      round(rain_mm, 2),
            "rain_flag":    rain_flag,
            "visibility_m": int(visibility_m),
            "fog_flag":     fog_flag,
            "wind_kmh":     round(wind_kmh, 1),
            "temp_c":       round(temp_c, 1),
            "condition":    condition,
        }

    except Exception as e:
        print(f"[ERROR] Weather API failed for {highway}: {e}")

        # ✅ Safe fallback (ensures system never breaks)
        return {
            "highway": highway,
            "rain_mm": 0,
            "rain_flag": 0,
            "visibility_m": 10000,
            "fog_flag": 0,
            "wind_kmh": 5,
            "temp_c": 25,
            "condition": "Clear"
        }


def get_condition_label(code: int) -> str:
    if code == 0:             return "Clear"
    elif code in range(1, 4): return "Partly cloudy"
    elif code in range(45,48):return "Foggy"
    elif code in range(51,68):return "Drizzle/Rain"
    elif code in range(71,78):return "Snow"
    elif code in range(80,83):return "Heavy Rain"
    elif code in range(95,100):return "Thunderstorm"
    else:                     return "Cloudy"


def get_all_weather() -> pd.DataFrame:
    rows = [get_weather(hw) for hw in HIGHWAYS]
    df = pd.DataFrame(rows)

    print("\nLive weather fetched for all highways:")
    print(df[["highway", "condition", "rain_mm",
              "visibility_m", "wind_kmh"]].to_string(index=False))

    return df


if __name__ == "__main__":
    df = get_all_weather()