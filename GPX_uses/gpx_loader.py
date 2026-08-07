import gpxpy
import pandas as pd
import srtm
import requests
from typing import Tuple, Union


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


def load_gpx_from_url(source: Union[str, bytes]) -> Tuple[gpxpy.gpx.GPX, str]:
    """
    Carga un GPX desde una URL (string) o desde contenido en memoria (bytes).

    - Si `source` es str, se asume que es una URL y se hace un requests.get().
    - Si `source` son bytes, se parsea directamente el contenido GPX.
    """
    if isinstance(source, str):
        response = requests.get(source)
        response.raise_for_status()
        gpx = gpxpy.parse(response.content)
    else:
        # Se asume que es contenido GPX en bytes (por ejemplo, un archivo subido)
        gpx = gpxpy.parse(source)

    name = gpx.name
    return gpx, name


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
