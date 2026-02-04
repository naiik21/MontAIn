import os
import json
import numpy as np
import pandas as pd
import Fetch.osm_fetch as osm
import Fetch.osm_to_gpx as gpx
from GPX_uses.gpx_loader import load_gpx
import datasetter
from GPX_uses.gpx_reloader import reloader_gpx
from NeuronalNetwork import model_training



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
            gpx.osm_relation_to_gpx(relation, nodes_dict, ways_dict, path)
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
    # Crear el dataset
    dataset = datasetter.build_dataset(gpx_dir=gpx_dir)
    
    # Guardar en CSV
    dataset.to_csv(output_file, index=False, encoding='utf-8')
    
    return dataset



def main():  
    bboxes = [
        ("pirineos", (42.2, 0.5, 43.3, 2.3)),
        ("picos_europa", (43.1, -5.0, 43.3, -4.7)),
        ("sierra_nevada", (36.9, -3.5, 37.1, -3.3)),
        ("gredos", (40.2, -5.3, 40.4, -5.0)),
        ("teide", (28.2, -16.7, 28.3, -16.6)),
        ("montseny", (41.7, 2.3, 41.8, 2.5)),
        ("alpes", (45.8, 6.8, 46.0, 7.0)),
        ("himalaya", (27.9, 86.8, 28.1, 87.0)),
        ("montanas_rocosas", (39.0, -106.5, 39.5, -105.5)),
        ("andes", (-33.0, -70.0, -32.5, -69.5)),
        ("kilimanjaro", (-3.1, 37.3, -3.0, 37.4)),
        ("japan", (35.3, 138.7, 35.4, 138.8)),
        ("matterhorn", (45.9, 7.6, 46.0, 7.7)),
        ("atlas", (31.0, -8.0, 31.5, -7.5)),
    ]

    for name, bbox in bboxes:
        print(f"\nDescargando rutas: {name}")
        osm_data = osm.fetch_hiking_routes(bbox)
        print(f"📦 Recibidos {len(osm_data['elements'])} elementos")

        print("Guardando rutas como GPX")
        save_all_routes(osm_data, output_dir=f"data/gpx/{name}")
        

    # print("\nCreando CSV del dataset")
    # save_dataset_to_csv(gpx_dir="data/gpx", output_file="dataset.csv")
    
    # print("\nEntrenando modelo")
    # model_training(batch_size=128, epochs=500, lr=0.001)
   
    # print("\nModelo entrenado")
    print("\nProceso finalizado")
if __name__ == "__main__":
    main()