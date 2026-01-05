import gpxpy
import pandas as pd
import srtm


def load_gpx(path):
    with open(path, "r") as f:
        gpx = gpxpy.parse(f)
        name = gpx.name
 
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append([point.latitude, point.longitude, point.elevation])

    return pd.DataFrame(data, columns=["lat", "lon", "ele"]), name
