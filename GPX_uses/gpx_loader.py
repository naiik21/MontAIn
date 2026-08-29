import gpxpy
import pandas as pd


def load_gpx(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
        name = gpx.name

    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append([point.latitude, point.longitude, point.elevation])

    return pd.DataFrame(data, columns=["lat", "lon", "ele"]), name


def gpx_to_dataframe(gpx_path):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "elevation": point.elevation
                })

    df = pd.DataFrame(points)
    return df
