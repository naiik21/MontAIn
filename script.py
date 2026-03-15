"""
Script para procesar un archivo GPX y obtener todos los datos de la ruta.
Uso: python process_gpx.py <ruta_al_archivo.gpx>
"""
import sys
import json
import datasetter
from api import map as route_map


def process_gpx(gpx_path: str) -> dict:
    df = datasetter.build_dataset_from_file(gpx_path)
    records = df.to_dict(orient="records")

    map_html = route_map.get_map(gpx_path)
    elevation_plot = route_map.get_elevation(gpx_path)

    return {
        "data": records,
        "map_html": map_html,
        "elevation_plot": elevation_plot,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python process_gpx.py <ruta_al_archivo.gpx>")
        sys.exit(1)

    gpx_path = sys.argv[1]
    result = process_gpx(gpx_path)

    # Imprimir datos de la ruta (sin el HTML del mapa)
    print(json.dumps(result["data"], indent=2, ensure_ascii=False))
    print(f"\nelevation_plot keys: {list(result['elevation_plot'].keys())}")
    print(f"map_html length: {len(result['map_html'])} chars")
