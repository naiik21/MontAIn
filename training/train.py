"""Pipeline de entrenamiento de MontAIn.

Los tres pasos se ejecutan en orden y cada uno consume la salida del anterior:

  1. fetch    Descarga rutas de senderismo de OpenStreetMap y las guarda como GPX
  2. dataset  Convierte esos GPX en un CSV de features + etiqueta de dificultad
  3. train    Entrena y evalua los modelos sobre ese CSV

Ejecutar siempre desde la raiz del repositorio:

  python -m training.train fetch --region pirineos
  python -m training.train dataset --gpx-dir data/gpx --out bigDataset.csv
  python -m training.train train --model xgboost-reg --save xgboost_regresion.json

NOTA SOBRE LAS ETIQUETAS: `dataset` etiqueta cada ruta con datasetter.classify_difficulty(),
un sistema de reglas escrito a mano. Los modelos entrenados aqui aprenden a reproducir esas
reglas, asi que sus metricas miden fidelidad a la heuristica, no acierto sobre dificultad real.
"""

import argparse
import os

import Fetch.osm_fetch as osm
import Fetch.osm_to_gpx as osm_gpx
import datasetter

# bbox = (sur, oeste, norte, este)
REGIONS = {
    "pirineos": (42.2, 0.5, 43.3, 2.3),
    "picos_europa": (43.1, -5.0, 43.3, -4.7),
    "sierra_nevada": (36.9, -3.5, 37.1, -3.3),
    "gredos": (40.2, -5.3, 40.4, -5.0),
    "teide": (28.2, -16.7, 28.3, -16.6),
    "montseny": (41.7, 2.3, 41.8, 2.5),
    "alpes": (45.8, 6.8, 46.0, 7.0),
    "himalaya": (27.9, 86.8, 28.1, 87.0),
    "montanas_rocosas": (39.0, -106.5, 39.5, -105.5),
    "andes": (-33.0, -70.0, -32.5, -69.5),
    "kilimanjaro": (-3.1, 37.3, -3.0, 37.4),
    "matterhorn": (45.9, 7.6, 46.0, 7.7),
    "atlas": (31.0, -8.0, 31.5, -7.5),
}


def save_all_routes(osm_data, output_dir="data/gpx"):
    """
    Guarda todas las rutas como archivos GPX individuales
    """
    os.makedirs(output_dir, exist_ok=True)

    # Crear diccionarios de nodos y ways para acceso rápido
    nodes_dict = {}
    ways_dict = {}
    relations = []

    for element in osm_data["elements"]:
        elem_type = element["type"]
        elem_id = element["id"]

        if elem_type == "node":
            nodes_dict[elem_id] = element
        elif elem_type == "way":
            ways_dict[elem_id] = element
        elif elem_type == "relation":
            relations.append(element)

    # Procesar cada relación
    count = 0
    for relation in relations:
        route_id = relation["id"]
        route_name = relation.get("tags", {}).get("name", f"route_{route_id}")

        # Sanitizar nombre para archivo
        safe_name = "".join(c for c in route_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name[:50]  # Limitar longitud

        path = f"{output_dir}/{safe_name}_{route_id}.gpx"

        try:
            osm_gpx.osm_relation_to_gpx(relation, nodes_dict, ways_dict, path)
            count += 1
            print(f"✓ Guardada: {route_name}")
        except Exception as e:
            print(f"✗ Error en {route_name}: {e}")

    print(f"\n🎉 Guardadas {count} rutas en {output_dir}/")
    return count


def save_dataset_to_csv(gpx_dir="data/gpx", output_file="dataset.csv"):
    """
    Crea el dataset y lo guarda en un archivo CSV
    """
    dataset = datasetter.build_dataset(gpx_dir=gpx_dir)
    dataset.to_csv(output_file, index=False, encoding='utf-8')
    return dataset


def cmd_fetch(args):
    bbox = REGIONS[args.region]
    print(f"Descargando rutas de '{args.region}' {bbox}")
    osm_data = osm.fetch_hiking_routes(bbox)
    print(f"📦 Recibidos {len(osm_data['elements'])} elementos")
    save_all_routes(osm_data, output_dir=args.gpx_dir)


def cmd_dataset(args):
    print(f"Construyendo dataset desde {args.gpx_dir}")
    dataset = save_dataset_to_csv(gpx_dir=args.gpx_dir, output_file=args.out)
    print(f"✓ {len(dataset)} rutas escritas en {args.out}")


def cmd_train(args):
    # Importados aqui dentro para que `fetch` y `dataset` no exijan
    # las dependencias pesadas de entrenamiento (sklearn, torch, seaborn).
    if args.model == "baseline":
        from training.models.baseline import baseline
        baseline(dataset_path=args.dataset, show_plots=args.plots)
    elif args.model == "xgboost":
        from training.models.xgboost_model import xgboost
        xgboost(dataset_path=args.dataset, show_plots=args.plots)
    elif args.model == "xgboost-reg":
        from training.models.xgboost_regresion import xgboost_regresion
        xgboost_regresion(dataset_path=args.dataset, save_path=args.save)
    elif args.model == "nn":
        from training.models.neural_network import model_training
        model_training(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Descarga rutas de OSM y las guarda como GPX")
    p_fetch.add_argument("--region", choices=sorted(REGIONS), required=True)
    p_fetch.add_argument("--gpx-dir", default="data/gpx")
    p_fetch.set_defaults(func=cmd_fetch)

    p_dataset = sub.add_parser("dataset", help="Convierte los GPX en un CSV de features")
    p_dataset.add_argument("--gpx-dir", default="data/gpx")
    p_dataset.add_argument("--out", default="bigDataset.csv")
    p_dataset.set_defaults(func=cmd_dataset)

    p_train = sub.add_parser("train", help="Entrena y evalua un modelo sobre el CSV")
    p_train.add_argument("--model", choices=["baseline", "xgboost", "xgboost-reg", "nn"], default="xgboost-reg")
    p_train.add_argument("--dataset", default="bigDataset.csv")
    p_train.add_argument("--save", default=None, help="Ruta donde guardar el modelo entrenado (solo xgboost-reg)")
    p_train.add_argument("--plots", action="store_true", help="Muestra las graficas de evaluacion")
    p_train.add_argument("--batch-size", type=int, default=32, help="Solo para --model nn")
    p_train.add_argument("--epochs", type=int, default=50, help="Solo para --model nn")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Solo para --model nn")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
