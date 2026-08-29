"""
Analiza una traza GPX desde la linea de comandos, sin levantar la API.

Util para probar el proyecto recien clonado: no necesita el frontend ni una
clave de Anthropic. Devuelve las mismas metricas y la misma estimacion de
dificultad que /process-gpx, pero no genera la descripcion con IA.

    python analyze_gpx.py front/public/ejemplos/almanzor.gpx
    python analyze_gpx.py mi-ruta.gpx --json      # salida completa, para pipes
"""

import argparse
import json
import sys

import datasetter
from api import map as route_map


def analyze(gpx_path: str) -> dict:
    df = datasetter.build_dataset_from_file(gpx_path)
    return {
        "data": df.to_dict(orient="records"),
        "map_html": route_map.get_map(gpx_path),
        "elevation_plot": route_map.get_elevation(gpx_path),
    }


def num(value, digits=0):
    # Formato espanol: punto para los miles y coma para los decimales.
    # translate intercambia ambos separadores en una sola pasada.
    return f"{value:,.{digits}f}".translate(str.maketrans(",.", ".,"))


def print_summary(result: dict) -> None:
    r = result["data"][0]
    profile = result["elevation_plot"]

    print(f"\n  {r.get('filename') or 'Ruta sin nombre'}")
    print(f"  {'-' * 46}")
    print(f"  Dificultad estimada   {r['difficulty']}")
    print(f"  Distancia             {num(r['distance_km'], 1)} km")
    print(f"  Desnivel +/-          {num(r['elevation_gain'])} / {num(r['elevation_loss'])} m")
    print(f"  Cotas min/max         {num(r['min_elevation'])} / {num(r['max_elevation'])} m")
    print(f"  Pendiente media/max   {num(r['mean_slope'], 1)} / {num(r['max_slope'], 1)} grados")
    print(f"  Tramo > 30 grados     {num(r['pct_over_30'] * 100, 1)} %")
    print(f"  Puntos del perfil     {num(len(profile['distance_km']))}")
    print(f"  Mapa generado         {num(len(result['map_html']))} caracteres")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("gpx", help="Ruta al archivo .gpx")
    parser.add_argument("--json", action="store_true", help="Imprime el resultado completo en JSON")
    args = parser.parse_args()

    try:
        result = analyze(args.gpx)
    except FileNotFoundError:
        print(f"No se encuentra el archivo: {args.gpx}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"No se ha podido analizar la traza: {e}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
