import hashlib
import logging
import os
import tempfile

import gpxpy
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
import datasetter
import api.map as map
import description.route_guide as route_guide
import description.build_compute as build_compute
import description.detection_events as detection_events
import description.key_moments as key_moment
import description.anthropic_claude as anthropic_claude
from api.cache import LRUCache
from api.ratelimit import SlidingWindowRateLimiter, client_ip
from GPX_uses.gpx_loader import gpx_to_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("montain")


class GPXValidationError(Exception):
    """El archivo subido no sirve. El mensaje es seguro de mostrar al cliente."""


# Inicializar FastAPI
app = FastAPI(
    title="MontAIn API",
    description="API para análisis de rutas de montaña usando IA",
    version="1.0.0",
)

# Configurar CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = SlidingWindowRateLimiter(
    max_requests=config.RATE_LIMIT_REQUESTS,
    window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
)

analysis_cache = LRUCache(max_entries=config.ANALYSIS_CACHE_SIZE)


@app.get("/")
async def root():
    """Endpoint raíz sencillo"""
    return {"message": "MontAIn API en ejecución"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "description_enabled": config.DESCRIPTION_ENABLED,
    }


def _validate_gpx(path: str) -> None:
    """
    Comprueba que el archivo es un GPX parseable y de tamano razonable.

    Se hace antes del analisis porque compute_distances calcula una distancia
    geodesica por cada par de puntos consecutivos: una traza con cientos de
    miles de puntos bloquearia el worker durante minutos.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    except Exception:
        raise GPXValidationError("El archivo no es un GPX válido o está corrupto.")

    points = sum(len(segment.points) for track in gpx.tracks for segment in track.segments)

    if points == 0:
        raise GPXValidationError(
            "El GPX no contiene ningún punto de track. "
            "Comprueba que el archivo incluya una traza y no solo waypoints."
        )

    if points > config.MAX_TRACK_POINTS:
        raise GPXValidationError(
            f"La traza tiene {points:,} puntos y el límite son "
            f"{config.MAX_TRACK_POINTS:,}. Simplifícala antes de subirla."
        )


def _build_description(record: dict, tmp_path: str) -> str | None:
    """
    Genera la descripción de la ruta con Claude.

    Devuelve None si no hay API key configurada o si la llamada falla: la
    descripción es un extra, y perderla no debe tumbar el análisis entero.
    """
    if not config.DESCRIPTION_ENABLED:
        return None

    try:
        df_points = gpx_to_dataframe(tmp_path)
        df_points = build_compute.build_features(df_points)
        events = detection_events.extract_route_events(df_points)
        key_moment.compute_fatigue_score(df_points)
        key_moments = key_moment.detect_key_moments(df_points)

        route_dict = {k: v for k, v in record.items() if k not in ("filename", "difficulty")}
        guide = route_guide.generate_route_guide(route_dict, [record["difficulty"]])

        prompt = f"""
Eres un guía de montaña profesional.
Redacta una descripción clara, responsable y realista de una ruta de montaña usando ÚNICAMENTE la información del JSON proporcionado.

Normas:
- No inventes información.
- No añadas pasos técnicos que no estén indicados.
- No menciones escalada salvo que se indique explícitamente.
- Usa un tono informativo, no épico.
- Prioriza la seguridad y la claridad.
- No hagas recomendaciones médicas ni técnicas avanzadas.

JSON:
{guide}

Eventos detectados:
{events}

Momentos clave:
{key_moments}
"""
        return anthropic_claude.generate_description(prompt)
    except Exception:
        logger.exception("Falló la generación de la descripción")
        return None


@app.post("/process-gpx")
async def process_gpx(request: Request, file: UploadFile = File(...)):
    """
    Recibe un archivo GPX subido, lo procesa con build_dataset_from_file
    y devuelve el resultado como JSON (una sola ruta).
    """
    if not file.filename or not file.filename.lower().endswith(".gpx"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un GPX (.gpx)")

    contents = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo supera el límite de {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # La cache se consulta antes que el rate limit: servir un resultado ya
    # calculado no cuesta CPU ni una llamada a Claude, asi que no gasta cuota.
    # En una demo publica lo habitual es que mucha gente pruebe con los mismos
    # archivos de ejemplo.
    file_hash = hashlib.sha256(contents).hexdigest()
    cached = analysis_cache.get(file_hash)
    if cached is not None:
        # Si la entrada se cacheo sin API key (description=None) y ahora hay
        # clave configurada, se reprocesa para completarla.
        if cached["description"] is not None or not config.DESCRIPTION_ENABLED:
            return JSONResponse(content=cached, headers={"X-Montain-Cache": "hit"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        _validate_gpx(tmp_path)

        # El rate limit se comprueba despues de validar, no antes: rechazar un
        # archivo no cuesta nada al servidor, asi que un usuario que se
        # equivoca de fichero no debe gastar cuota. Lo que se limita es el
        # trabajo caro que viene a continuacion (pipeline + llamada a Claude).
        allowed, retry_after = limiter.check(client_ip(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Has alcanzado el límite de {config.RATE_LIMIT_REQUESTS} rutas por hora. "
                    f"Vuelve a intentarlo en {retry_after // 60 + 1} minutos."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        map_html = map.get_map(tmp_path)
        elevation_plot = map.get_elevation(tmp_path)

        # Features agregadas + dificultad
        df = datasetter.build_dataset_from_file(tmp_path)
        records = df.to_dict(orient='records')
        record = records[0]

        description = _build_description(record, tmp_path)

        payload = {
            'data': records,
            'map_html': map_html,
            'elevation_plot': elevation_plot,
            'description': description,
        }
        analysis_cache.put(file_hash, payload)

        return JSONResponse(content=payload, headers={"X-Montain-Cache": "miss"})

    except GPXValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        # El traceback va al log del servidor, no al cliente: los mensajes de
        # excepcion filtran rutas del sistema de archivos y detalles internos.
        logger.exception("Error procesando %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail="No se ha podido procesar la ruta. Inténtalo de nuevo en unos minutos.",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    print("🚀 Iniciando servidor MontAIn API...")
    print(f"📡 Servidor disponible en http://localhost:{config.PORT}")
    print(f"📚 Documentación en http://localhost:{config.PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
