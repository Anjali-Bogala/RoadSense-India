import pandas as pd
import random

def generate_accident_data(n=200):
    """
    Generate realistic accident data with state info
    """

    data = []

    # Highway regions (Jaipur → Delhi route)
    locations = [
        (26.9124, 75.7873, "Rajasthan"),  # Jaipur
        (27.2, 76.2, "Rajasthan"),
        (27.6, 76.8, "Haryana"),
        (28.1, 77.0, "Haryana"),
        (28.6139, 77.2090, "Delhi")       # Delhi
    ]

    for _ in range(n):
        base_lat, base_lon, state = random.choice(locations)

        lat = base_lat + random.uniform(-0.05, 0.05)
        lon = base_lon + random.uniform(-0.05, 0.05)

        severity = random.randint(1, 5)

        data.append({
            "latitude": lat,
            "longitude": lon,
            "severity": severity,
            "state": state
        })

    df = pd.DataFrame(data)
    df.to_csv("data/accidents_clean.csv", index=False)

    print("✅ Accident data generated → data/accidents_clean.csv")