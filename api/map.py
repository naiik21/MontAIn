import logging
import os
import tempfile

import gpxpy
import folium
import srtm

import config

logger = logging.getLogger("montain.map")

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

    # El mapa ocupa todo el iframe; el encuadre se ajusta a la traza al final.
    mapa = folium.Map(location=coords[0], zoom_start=10, width='100%', height='100%')
    
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
    folium.PolyLine(coords, color='#c13a2a', weight=3.5, opacity=0.9).add_to(mapa)

    # Sin esto el documento hereda el margen por defecto del body y html/body
    # no tienen altura, asi que Leaflet se inicializa midiendo mal su
    # contenedor: el iframe saca barra de scroll y fit_bounds encuadra sobre
    # un tamano equivocado (se acaba viendo la region entera, no la traza).
    mapa.get_root().header.add_child(folium.Element(
        "<style>html,body{height:100%;margin:0;padding:0;overflow:hidden}"
        ".folium-map{position:absolute;inset:0}</style>"
    ))

    # Encuadrar la traza completa en vez de dejar un zoom fijo arbitrario.
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    mapa.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))

    # Documento HTML completo, listo para el srcdoc de un iframe.
    # _repr_html_() no sirve aqui: esta pensado para Jupyter y devuelve un
    # iframe anidado mas un aviso de "Make this Notebook Trusted", que dentro
    # de nuestro iframe con sandbox impide que Leaflet arranque.
    map_html = mapa.get_root().render()

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
        _elevation_data = srtm.get_data(
            local_cache_dir=_usable_cache_dir(),
            timeout=config.SRTM_TIMEOUT_SECONDS,
        )
    return _elevation_data


def _usable_cache_dir() -> str:
    """
    Devuelve SRTM_CACHE_DIR si se puede escribir en el; si no, un temporal.

    En un despliegue con disco persistente, el volumen se monta ENCIMA del
    directorio de la imagen y trae su propia propiedad, que puede no coincidir
    con el usuario del contenedor. Sin esta comprobacion, el primer intento de
    descargar un tile tumbaria la peticion. Cayendo a un temporal la API sigue
    funcionando: los tiles se redescargan en cada arranque, que es lento pero
    no es una caida.
    """
    path = config.SRTM_CACHE_DIR
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".escritura")
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)
        return path
    except OSError as e:
        fallback = os.path.join(tempfile.gettempdir(), "montain-srtm")
        os.makedirs(fallback, exist_ok=True)
        logger.warning(
            "No se puede escribir en SRTM_CACHE_DIR (%s): %s. "
            "Usando %s; los tiles se redescargaran en cada arranque.",
            path, e, fallback,
        )
        return fallback


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