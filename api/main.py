from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import gpxpy

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],  # Tu app Astro
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-gpx")
async def process_gpx(file: UploadFile = File(...)):
    content = await file.read()
    gpx = gpxpy.parse(content)
    
    # Aquí procesas con tu modelo IA
    # resultado = tu_modelo_ia.predict(gpx)
    
    return {
        "tracks": len(gpx.tracks),
        "points": sum(len(track.segments[0].points) for track in gpx.tracks),
        # tus datos procesados
    }