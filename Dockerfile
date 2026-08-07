FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Las dependencias van en su propia capa: mientras requirements.txt no cambie,
# reconstruir la imagen tras editar codigo no reinstala nada.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Directorio por defecto de la cache de tiles SRTM. Montar aqui un volumen
# persistente evita volver a descargarlos en cada arranque en frio.
ENV SRTM_CACHE_DIR=/app/.srtm-cache
RUN mkdir -p /app/.srtm-cache

# Usuario sin privilegios; necesita poder escribir en la cache.
RUN useradd --create-home --uid 1000 montain \
    && chown -R montain:montain /app
USER montain

EXPOSE 8000

# $PORT lo inyecta el proveedor (Render, Railway...); 8000 en local.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
