# MontAIn

Subes la traza GPX de una ruta de montaña y MontAIn te devuelve una ficha: métricas
del recorrido, mapa, perfil de elevación, una estimación de dificultad hecha por un
modelo entrenado sobre rutas de OpenStreetMap y una descripción escrita por Claude.

![Pantalla de inicio de MontAIn](docs/captura-inicio.png)

## La ficha de ruta

El color codifica datos en vez de decorar. En el perfil de elevación el relleno toma
el color de la altitud, como las tintas hipsométricas de un mapa topográfico: esta
ruta al Almanzor arranca en el verde de los 1193 m y termina en el gris de roca de
los 2523 m. Dos rutas a distinta cota se distinguen antes de leer una sola cifra.

![Ficha de ruta con mapa, perfil de elevación, métricas y descripción](docs/captura-ficha.png)

## Prueba rápida

Lo mínimo para ver el análisis funcionando. No necesita el frontend ni clave de API:

```bash
git clone https://github.com/naiik21/MontAIn.git
cd MontAIn

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python analyze_gpx.py front/public/ejemplos/almanzor.gpx
```

```
  Almanzor por el Camino del Tío Domingo
  ----------------------------------------------
  Dificultad estimada   alpinismo ligero
  Distancia             8,2 km
  Desnivel +/-          1.789 / 491 m
  Cotas min/max         1.193 / 2.523 m
  Pendiente media/max   8,8 / 84,8 grados
  Tramo > 30 grados     14,0 %
```

Añade `--json` para la salida completa, con el mapa y el perfil, lista para redirigir
a un fichero o a otro programa.

En [`front/public/ejemplos/`](front/public/ejemplos/) hay dos trazas de muestra.
Cualquier `.gpx` sirve: puedes exportarlos de Wikiloc, Strava, Komoot o de tu reloj.

## La aplicación completa

Son dos procesos: la API en Python y el frontend en Astro. Necesitas Python 3.11 y
Node 20.

### 1. La API

Con el entorno ya creado en el paso anterior:

```bash
cp .env.example .env               # Windows: copy .env.example .env
python main.py
```

Arranca en `http://localhost:8000`, con la documentación interactiva en
`http://localhost:8000/docs`.

La descripción con IA es opcional. Si quieres generarla, pon tu clave de
[console.anthropic.com](https://console.anthropic.com) en `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Sin clave todo lo demás funciona igual y el campo `description` viene a `null`.

### 2. El frontend

En otra terminal:

```bash
cd front
pnpm install                       # o npm install
pnpm dev                           # o npm run dev
```

Abre `http://localhost:4321`. Si la API no está en `localhost:8000`, copia
`front/.env.example` a `front/.env` y define ahí `PUBLIC_API_URL`.

## Configuración

Todo tiene un valor por defecto pensado para desarrollo, así que la API arranca sin
configurar nada. [`.env.example`](.env.example) documenta cada variable; estas son
las que más se tocan:

| Variable | Por defecto | Para qué |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Sin ella no se generan descripciones |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Modelo que redacta la descripción |
| `CORS_ORIGINS` | localhost | Dominios que pueden llamar a la API |
| `MAX_UPLOAD_BYTES` | 5 MB | Tamaño máximo del GPX |
| `MAX_TRACK_POINTS` | 50 000 | Puntos máximos de la traza |
| `RATE_LIMIT_REQUESTS` | 10 / hora | Peticiones por IP |
| `ANALYSIS_CACHE_SIZE` | 128 | Resultados cacheados por hash del archivo |
| `SRTM_CACHE_DIR` | `.srtm-cache` | Dónde se guardan los tiles de elevación |

## Cómo funciona

Una petición a `POST /process-gpx` recorre este camino:

| Paso | Qué hace |
|---|---|
| Validación | Extensión, tamaño y número de puntos de la traza |
| Elevación | Rellena las cotas que falten con datos SRTM |
| Features | Distancia geodésica, pendientes, rugosidad, exposición y orientación por tramo |
| Dificultad | Un XGBoost de regresión sobre 13 features, redondeado a seis grados ordinales |
| Mapa | Folium sobre OpenStreetMap, servido en un iframe con sandbox |
| Descripción | Claude redacta una guía a partir de las features, los eventos del recorrido y los momentos clave |

Los resultados se cachean por hash del archivo: el mismo GPX no se analiza ni se envía
a Claude dos veces.

## Sobre la estimación de dificultad

El modelo se entrena con etiquetas que genera `classify_difficulty()`, un sistema de
reglas escrito a mano. Es decir, **aprende a reproducir esa heurística**, no a predecir
la dificultad real: sus métricas miden fidelidad a las reglas, no acierto. Se nota en
los falsos positivos —hay paseos suaves que salen como "difícil"— y corregirlo pasa por
reetiquetar el dataset con criterio humano o con una fuente externa. Úsalo como una
orientación, no como un grado oficial.

## Si algo falla

| Síntoma | Causa |
|---|---|
| `No se ha podido contactar con el servidor` en el navegador | La API no está arrancada, o `PUBLIC_API_URL` apunta a otro sitio |
| La primera petición tarda muchísimo | SRTM está descargando los tiles de elevación de la zona. Se cachean en `SRTM_CACHE_DIR` y la siguiente ya es rápida |
| `description` siempre viene a `null` | Falta `ANTHROPIC_API_KEY` en `.env`. Compruébalo en `/health` |
| `El GPX no contiene ningún punto de track` | El archivo solo tiene waypoints, sin traza |
| `429` al probar varias rutas seguidas | El límite por IP. Sube `RATE_LIMIT_REQUESTS` en `.env` |
| La descripción se corta a media frase | `ANTHROPIC_MAX_TOKENS` demasiado bajo |

## Despliegue

La API va en un contenedor ([`Dockerfile`](Dockerfile)) y el frontend en Vercel con
`front/` como raíz.

```bash
docker build -t montain-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... montain-api
```

[`render.yaml`](render.yaml) despliega la API en Render con un disco persistente para
los tiles SRTM. Sin ese disco, cada arranque en frío vuelve a descargarlos y la primera
petición se dispara.

Los dos servicios se apuntan mutuamente: `CORS_ORIGINS` en la API con el dominio de
Vercel, y `PUBLIC_API_URL` en Vercel con el de Render.

## Entrenamiento

El modelo desplegado (`xgboost_regresion.json`) ya está en el repositorio. Para
regenerarlo:

```bash
pip install -r training/requirements.txt

python -m training.train fetch --region pirineos
python -m training.train dataset --out bigDataset.csv
python -m training.train train --model xgboost-reg --save xgboost_regresion.json
```

Hay cuatro modelos comparables (`baseline`, `xgboost`, `xgboost-reg`, `nn`) en
[`training/models/`](training/models/).

## Estructura

```
main.py              API FastAPI
analyze_gpx.py       CLI para analizar un GPX sin levantar la API
config.py            Configuración por variables de entorno
datasetter.py        Features del recorrido y estimación de dificultad
api/                 Mapa, rate limiting y caché
description/         Pipeline de la descripción con IA
GPX_uses/, Fetch/    Carga de GPX y descarga desde OpenStreetMap
training/            Pipeline de entrenamiento
front/               Frontend Astro + React
integrations/n8n/    Workflow que publica la ficha en WordPress
```

## Stack

FastAPI · XGBoost · pandas · Folium · SRTM · Astro · React · Claude

## Licencia

[MIT](LICENSE)
