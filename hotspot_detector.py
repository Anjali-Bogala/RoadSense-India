import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pickle, os

def load_accident_data() -> pd.DataFrame:
    df = pd.read_csv("data/accidents_clean.csv")

    # Make sure lat/lon exist
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower()]

    if not lat_cols or not lon_cols:
        print("No lat/lon found — using state centroids")
        STATE_COORDS = {
            "Rajasthan":       (27.0238, 74.2179),
            "Maharashtra":     (19.7515, 75.7139),
            "Uttar Pradesh":   (26.8467, 80.9462),
            "Tamil Nadu":      (11.1271, 78.6569),
            "Karnataka":       (15.3173, 75.7139),
            "Gujarat":         (22.2587, 71.1924),
            "Madhya Pradesh":  (22.9734, 78.6569),
            "Bihar":           (25.0961, 85.3131),
            "West Bengal":     (22.9868, 87.8550),
            "Andhra Pradesh":  (15.9129, 79.7400),
            "Telangana":       (17.1232, 79.2088),
            "Punjab":          (31.1471, 75.3412),
            "Haryana":         (29.0588, 76.0856),
            "Odisha":          (20.9517, 85.0985),
            "Jharkhand":       (23.6102, 85.2799),
        }
        state_col = [c for c in df.columns if "state" in c.lower()][0]
        df["latitude"]  = df[state_col].map(
            lambda s: STATE_COORDS.get(str(s).strip(),
                                       (20.5937, 78.9629))[0])
        df["longitude"] = df[state_col].map(
            lambda s: STATE_COORDS.get(str(s).strip(),
                                       (20.5937, 78.9629))[1])
    else:
        df = df.rename(columns={lat_cols[0]: "latitude",
                                 lon_cols[0]: "longitude"})

    # Add small random jitter so points don't all stack
    df["latitude"]  += np.random.uniform(-0.3, 0.3, len(df))
    df["longitude"] += np.random.uniform(-0.3, 0.3, len(df))

    df = df.dropna(subset=["latitude", "longitude"])
    print(f"Loaded {len(df)} accident records for clustering")
    return df


def run_dbscan(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning DBSCAN clustering...")

    coords = df[["latitude", "longitude"]].values
    coords_rad = np.radians(coords)

    # haversine metric gives real geographic distance
    db = DBSCAN(
        eps=0.08,          # ~8km radius
        min_samples=3,     # minimum 3 accidents = hotspot
        algorithm="ball_tree",
        metric="haversine"
    )

    df = df.copy()
    df["cluster"] = db.fit_predict(coords_rad)

    total_clusters = df[df["cluster"] != -1]["cluster"].nunique()
    noise_points   = (df["cluster"] == -1).sum()

    print(f"Hotspot clusters found : {total_clusters}")
    print(f"Isolated points (noise): {noise_points}")
    print(f"Clustered accident pts : {len(df) - noise_points}")

    # Cluster size for coloring
    cluster_sizes = (df[df["cluster"] != -1]
                     .groupby("cluster")
                     .size()
                     .rename("cluster_size"))
    df = df.merge(cluster_sizes, on="cluster", how="left")
    df["cluster_size"] = df["cluster_size"].fillna(0).astype(int)

    df.to_csv("data/clustered_accidents.csv", index=False)
    print("\nSaved → data/clustered_accidents.csv")
    return df


def get_hotspot_summary(df: pd.DataFrame) -> pd.DataFrame:
    hotspots = df[df["cluster"] != -1].copy()
    summary  = (hotspots.groupby("cluster")
                .agg(
                    accident_count=("cluster", "count"),
                    center_lat=("latitude", "mean"),
                    center_lon=("longitude", "mean"),
                )
                .reset_index()
                .sort_values("accident_count", ascending=False))

    max_count = summary["accident_count"].max()
    summary["risk_level"] = summary["accident_count"].apply(
        lambda x: "HIGH"   if x > max_count * 0.6 else
                  "MEDIUM" if x > max_count * 0.3 else "LOW")

    summary.to_csv("data/hotspot_summary.csv", index=False)
    print("\nTop 10 hotspots:")
    print(summary.head(10).to_string(index=False))
    return summary


if __name__ == "__main__":
    df  = load_accident_data()
    df  = run_dbscan(df)
    summary = get_hotspot_summary(df)
    print("\nDBSCAN complete!")