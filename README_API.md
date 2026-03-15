# MontAIn API - Guía de Uso

Este proyecto incluye un servidor API para analizar rutas de montaña usando modelos de Machine Learning.

## Estructura de Archivos

- `main.py` - **NUEVO**: Servidor API FastAPI para procesar archivos GPX
- `main_train.py` - **ANTIGUO**: Script para entrenar modelos (renombrado desde main.py)

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Entrenar el modelo (si aún no está entrenado)

Si no tienes el archivo `xgboost_regresion.model`, primero entrena el modelo:

```bash
python main_train.py
```

Esto generará el archivo `xgboost_regresion.model` necesario para hacer predicciones.

### 2. Iniciar el servidor API

```bash
python main.py
```

El servidor se iniciará en `http://localhost:8000`

### 3. Usar el frontend

En otra terminal, inicia el frontend:

```bash
cd front
pnpm install  # o npm install
pnpm dev     # o npm run dev
```

El frontend estará disponible en `http://localhost:4321` (puerto por defecto de Astro)

## Endpoints de la API

### GET `/`
Endpoint raíz que muestra información del estado de la API.

### GET `/health`
Health check para verificar que el servidor está funcionando.

### POST `/process-gpx`
Procesa un archivo GPX y devuelve el análisis completo.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Archivo GPX en el campo `file`

**Response:**
```json
{
  "route_name": "Nombre de la ruta",
  "metrics": {
    "distance_km": 18.5,
    "elevation_gain": 1350.0,
    "elevation_loss": 1200.0,
    "max_elevation": 3718.0,
    "min_elevation": 2368.0,
    "max_slope": 45.2,
    "mean_slope": 12.5
  },
  "risk_indicators": {
    "pct_over_30": 0.15,
    "pct_over_40": 0.08,
    "pct_over_45": 0.03,
    "exposed_pct": 0.12,
    "rugosity_mean": 8.5
  },
  "difficulty": {
    "level": 4,
    "name": "alpinismo ligero",
    "confidence": 0.85
  },
  "all_features": { ... }
}
```

## Documentación Interactiva

Una vez que el servidor esté corriendo, puedes acceder a la documentación interactiva de la API en:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Solución de Problemas

### Error: "Modelo no disponible"
- Asegúrate de que el archivo `xgboost_regresion.model` existe en el directorio raíz
- Si no existe, ejecuta `python main_train.py` para entrenar el modelo

### Error de CORS
- El servidor está configurado para aceptar requests desde `localhost:4321` (Astro)
- Si usas otro puerto, modifica la configuración CORS en `main.py`

### Error al procesar GPX
- Verifica que el archivo GPX sea válido
- Asegúrate de que el archivo contenga puntos GPS con coordenadas y elevación

## Desarrollo

Para modificar el servidor:
1. Edita `main.py`
2. Reinicia el servidor para aplicar los cambios
3. Los cambios en el frontend se recargan automáticamente con Astro

