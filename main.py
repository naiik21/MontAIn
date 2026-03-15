import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import datasetter 
import api.map as map


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
        # Usar el flujo existente de datasetter para un solo archivo
        df = datasetter.build_dataset_from_file(tmp_path)
        # Convertir DataFrame a lista de diccionarios (orient='records' devuelve una lista)
        records = df.to_dict(orient='records')
        
        # Devolver directamente la lista, FastAPI la serializa automáticamente

        return JSONResponse(content={
            'data': records,
            'map_html': map_html,
            'elevation_plot': elevation_plot
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
