import numpy as np

from geopy.distance import geodesic

def compute_distance(df):
    distances = [0]

    for i in range(1, len(df)):
        prev = (df.iloc[i-1]["lat"], df.iloc[i-1]["lon"])
        curr = (df.iloc[i]["lat"], df.iloc[i]["lon"])
        d = geodesic(prev, curr).km
        distances.append(distances[-1] + d)

    df["distance_km"] = distances
    return df

def compute_slope(df):
    slopes = [0]

    for i in range(1, len(df)):
        elev_diff = df.iloc[i]["elevation"] - df.iloc[i-1]["elevation"]
        dist_diff = (df.iloc[i]["distance_km"] - df.iloc[i-1]["distance_km"]) * 1000

        slope = (elev_diff / dist_diff) * 100 if dist_diff > 0 else 0
        slopes.append(slope)

    df["slope"] = slopes
    return df

def compute_bearing(df):
    bearings = [0]

    for i in range(1, len(df)):
        lat1 = np.radians(df.iloc[i-1]["lat"])
        lon1 = np.radians(df.iloc[i-1]["lon"])
        lat2 = np.radians(df.iloc[i]["lat"])
        lon2 = np.radians(df.iloc[i]["lon"])

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))
        bearings.append(bearing)

    df["bearing"] = bearings
    return df

def build_features(df):
    df = compute_distance(df)
    df = compute_slope(df)
    df = compute_bearing(df)
    return df

