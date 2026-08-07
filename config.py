"""Configuracion de la API, leida de variables de entorno.

Todos los valores tienen un default pensado para desarrollo en local, asi que
la API arranca sin configurar nada. En produccion se sobreescriben desde el
panel del proveedor (Render, Railway...). Ver .env.example.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Se carga aqui, y no en el punto de entrada, para que este modulo sea
# autosuficiente: si .env se cargara desde main.py, cualquier script que
# importara config antes que main leeria variables vacias.
load_dotenv(BASE_DIR / ".env")


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"La variable {name} debe ser un entero, no {raw!r}")


# --- Red -------------------------------------------------------------------

# Origenes permitidos por CORS. En produccion: el dominio del frontend.
CORS_ORIGINS = _csv(
    "CORS_ORIGINS",
    "http://localhost:4321,http://127.0.0.1:4321,http://localhost:3000",
)

PORT = _int("PORT", 8000)


# --- Limites del endpoint --------------------------------------------------

# Un GPX de una ruta de montana normal no llega a 1 MB. 5 MB deja margen de
# sobra para trazas muy densas sin permitir subidas abusivas.
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)

# El coste del pipeline crece con el numero de puntos: compute_distances hace
# un calculo geodesico por par consecutivo.
MAX_TRACK_POINTS = _int("MAX_TRACK_POINTS", 50_000)

# Rate limit por IP: ventana deslizante.
RATE_LIMIT_REQUESTS = _int("RATE_LIMIT_REQUESTS", 10)
RATE_LIMIT_WINDOW_SECONDS = _int("RATE_LIMIT_WINDOW_SECONDS", 3600)


# --- Datos de elevacion ----------------------------------------------------

# SRTM descarga tiles de elevacion bajo demanda. Sin un directorio persistente,
# cada arranque en frio vuelve a descargarlos y la primera peticion se dispara.
# En produccion apuntar a un volumen que sobreviva a los reinicios.
SRTM_CACHE_DIR = os.getenv("SRTM_CACHE_DIR", str(BASE_DIR / ".srtm-cache"))
SRTM_TIMEOUT_SECONDS = _int("SRTM_TIMEOUT_SECONDS", 30)


# --- IA --------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
ANTHROPIC_MAX_TOKENS = _int("ANTHROPIC_MAX_TOKENS", 1000)

# Permite desplegar la demo sin clave: el analisis funciona y la descripcion
# se omite, en lugar de que falle la peticion entera.
DESCRIPTION_ENABLED = bool(ANTHROPIC_API_KEY)
