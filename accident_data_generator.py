import pandas as pd
import random

def generate_accident_data(n=200):
    """
    Generate realistic accident data for highways
    """
    data = []

    # Sample highway coordinates (Jaipur–Delhi region)
    base_locations = [
        (26.9124, 75.7873),  # Jaipur
        (27.0, 76.0),
        (27.5, 76.5),
        (28.0, 77.0),
        (28.6139, 77.2090)   # Delhi
    ]

    for _ in range(n):
        base_lat, base_lon = random.choice(base_locations)

        lat = base_lat + random.uniform(-0.05, 0.05)
        lon = base_lon + random.uniform(-0.05, 0.05)

        severity = random.randint(1, 5)

        data.append({
            "latitude": lat,
            "longitude": lon,
            "severity": severity
        })

    df = pd.DataFrame(data)
    df.to_csv("data/accidents_clean.csv", index=False)

    print("✅ Accident data generated → data/accidents_clean.csv")