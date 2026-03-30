import pandas as pd
from iot_simulator import generate_batch, HIGHWAYS
from weather_api import get_all_weather

def build_merged_dataset() -> pd.DataFrame:
    print("Step 1: Generating IoT data...")
    iot_df = generate_batch(readings_per_highway=100)

    print("\nStep 2: Fetching live weather...")
    weather_df = get_all_weather()

    print("\nStep 3: Merging...")
    merged = iot_df.merge(weather_df, on="highway", how="left")

    # Feature: combined danger score inputs
    merged["speed_risk"]   = (100 - merged["speed_kmh"]).clip(0, 100) / 100
    merged["density_risk"] = merged["density_veh"] / 200
    merged["weather_risk"] = (merged["rain_flag"] * 0.4 +
                               merged["fog_flag"]  * 0.4 +
                               (merged["wind_kmh"] / 100) * 0.2)

    merged.to_csv("data/merged_data.csv", index=False)
    print(f"\nMerged dataset saved → data/merged_data.csv")
    print(f"Shape: {merged.shape}")
    print(merged[["highway","speed_kmh","density_veh",
                  "condition","rain_flag","fog_flag"]].head(10))
    return merged

if __name__ == "__main__":
    build_merged_dataset()