import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import datasetter
import api.map as map
import description.route_guide as route_guide
import description.build_compute as build_compute
import description.detection_events as detection_events
import description.key_moments as key_moment
import description.anthropic_claude as anthropic_claude
from GPX_uses.gpx_loader import gpx_to_dataframe


# Inicializar FastAPI
app = FastAPI(
    title="MontAIn API",
    description="API para análisis de rutas de montaña usando IA",
    version="1.0.0",
)

# Configurar CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://localhost:3000",
        "http://127.0.0.1:4321",
    ],  # Astro dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raíz sencillo"""
    return {"message": "MontAIn API en ejecución"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
    }


@app.post("/process-gpx")
async def process_gpx(file: UploadFile = File(...)):
    """
    Recibe un archivo GPX subido, lo procesa con build_dataset_from_file
    y devuelve el resultado como JSON (una sola ruta).
    """
    if not file.filename.endswith(".gpx"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser un GPX (.gpx)",
        )

    # Guardar el archivo en un temporal para pasarlo a build_dataset_from_file
    tmp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        map_html = map.get_map(tmp_path)
        elevation_plot = map.get_elevation(tmp_path)

        # Features agregadas + dificultad
        df = datasetter.build_dataset_from_file(tmp_path)
        records = df.to_dict(orient='records')
        record = records[0]

        # Pipeline de descripción con IA
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
        description = anthropic_claude.generate_description(prompt)

        return JSONResponse(content={
            'data': records,
            'map_html': map_html,
            'elevation_plot': elevation_plot,
            'description': description,
        })

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar el GPX: {e}",
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
    print("📡 Servidor disponible en http://localhost:8000")
    print("📚 Documentación en http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
