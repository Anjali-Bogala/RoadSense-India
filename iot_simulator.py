import numpy as np
import pandas as pd
import random
import datetime
from faker import Faker

fake = Faker("en_IN")

HIGHWAYS = {
    "NH-48 Jaipur-Delhi":    (26.9124, 75.7873),
    "NH-44 Delhi-Srinagar":  (28.6139, 77.2090),
    "NH-8 Mumbai-Pune":      (19.0760, 72.8777),
    "NH-66 Bangalore-Kochi": (12.9716, 77.5946),
    "NH-16 Chennai-Kolkata": (13.0827, 80.2707),
}

def simulate_reading(highway: str) -> dict:
    lat, lon = HIGHWAYS[highway]
    hour = datetime.datetime.now().hour
    # Rush hours (8-10am, 5-8pm) = higher density
    is_rush = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0

    speed   = round(np.random.normal(55 if is_rush else 75, 15), 1)
    density = random.randint(80, 200) if is_rush else random.randint(10, 80)
    incident = random.choices([0, 1], weights=[0.93, 0.07])[0]

    return {
        "highway":        highway,
        "latitude":       lat + round(random.uniform(-0.5, 0.5), 4),
        "longitude":      lon + round(random.uniform(-0.5, 0.5), 4),
        "timestamp":      datetime.datetime.now().isoformat(),
        "speed_kmh":      max(10, speed),
        "density_veh":    density,
        "incident_flag":  incident,
    }

def generate_batch(readings_per_highway: int = 100) -> pd.DataFrame:
    rows = []
    for hw in HIGHWAYS:
        for _ in range(readings_per_highway):
            rows.append(simulate_reading(hw))
    df = pd.DataFrame(rows)
    df.to_csv("data/iot_sim_data.csv", index=False)
    print(f"Generated {len(df)} IoT readings → data/iot_sim_data.csv")
    return df

if __name__ == "__main__":
    df = generate_batch(100)
    print(df.head())
    print("\nHighway distribution:")
    print(df["highway"].value_counts())
    
    
