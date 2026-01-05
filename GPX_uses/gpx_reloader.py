import numpy as np
import pandas as pd
from geopy.distance import geodesic
import gpxpy


def reloader_gpx(df, name):
    """
    Re-muestrea la ruta cada X metros
    """
    # Crear objeto GPX
    gpx = gpxpy.gpx.GPX()
    gpx.name = name
    # Crear track
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx_track.name = gpx.name
    gpx.tracks.append(gpx_track)
    
    # Crear segment
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    # Añadir puntos al segment
    for _, row in df.iterrows():
        gpx_segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=row["lat"],
                longitude=row["lon"],
                elevation=row["ele"]
            )
        )
    
    return gpx