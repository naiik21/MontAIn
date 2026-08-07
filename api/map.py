import os

import gpxpy
import folium
import srtm

import config

def get_map(gpx_file):
    # Leer el archivo GPX
    with open(gpx_file, 'r', encoding='utf-8') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    # Extraer coordenadas
    
    coords = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append([point.latitude, point.longitude])

    # Crear mapa centrado en la primera coordenada
    mapa = folium.Map(location=coords[0], zoom_start=10)
    
    folium.Marker(
        location=[coords[0][0], coords[0][1]],
        popup="Inicio",
        icon=folium.Icon(color="red", icon="play", prefix='fa'), 
    ).add_to(mapa)

    # Marcador de fin
    folium.Marker(
        location=[coords[-1][0], coords[-1][1]],  
        popup="Fin",
        icon=folium.Icon(color="green", icon="stop", prefix='fa'), 
    ).add_to(mapa)

    # Dibujar la ruta
    folium.PolyLine(coords, color='blue', weight=2.5, opacity=0.8).add_to(mapa)

    # Convertir mapa a HTML
    map_html = mapa._repr_html_()

    return map_html


_elevation_data = None


def get_elevation_data():
    """
    Devuelve el cliente SRTM, creandolo una sola vez.

    srtm.get_data() prepara el indice de tiles en cada llamada, y los tiles se
    descargan bajo demanda. Cachear el cliente y fijar local_cache_dir a un
    directorio persistente evita repetir ambas cosas en cada peticion.
    """
    global _elevation_data
    if _elevation_data is None:
        os.makedirs(config.SRTM_CACHE_DIR, exist_ok=True)
        _elevation_data = srtm.get_data(
            local_cache_dir=config.SRTM_CACHE_DIR,
            timeout=config.SRTM_TIMEOUT_SECONDS,
        )
    return _elevation_data


def get_elevation(gpx_file):
    # Leer archivo GPX
    with open(gpx_file, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)

    # Completar elevaciones usando SRTM (si faltan / están a 0)
    get_elevation_data().add_elevations(gpx, smooth=True)

    distancias = []
    elevaciones = []
    dist_total = 0.0

    for track in gpx.tracks:
        for segment in track.segments:
            punto_anterior = None
            for punto in segment.points:
                if punto_anterior:
                    # distancia 2D entre puntos, en km
                    dist_total += punto.distance_2d(punto_anterior) / 1000.0

                distancias.append(dist_total)
                elevaciones.append(punto.elevation)
                punto_anterior = punto

    # Devolvemos datos crudos; FastAPI los serializa bien a JSON
    return {
        "distance_km": distancias,
        "elevation_m": elevaciones,
    }