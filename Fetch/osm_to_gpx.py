import gpxpy
import srtm

def points_are_close(p1, p2, threshold=0.00001):
    """Verifica si dos puntos están muy cerca (mismo nodo)"""
    return (abs(p1["lat"] - p2["lat"]) < threshold and 
            abs(p1["lon"] - p2["lon"]) < threshold)


def get_way_points(way, nodes_dict, reverse=False):
    """Obtiene los puntos de un way, opcionalmente invertido"""
    way_points = []
    node_ids = way.get("nodes", [])
    
    if reverse:
        node_ids = node_ids[::-1]
    
    for node_id in node_ids:
        if node_id in nodes_dict:
            node = nodes_dict[node_id]
            way_points.append({
                "lat": node["lat"],
                "lon": node["lon"]
            })
    
    return way_points


def get_way_endpoints(way, nodes_dict):
    """Obtiene el primer y último nodo de un way"""
    node_ids = way.get("nodes", [])
    if not node_ids:
        return None, None
    
    first_node_id = node_ids[0]
    last_node_id = node_ids[-1]
    
    first_node = nodes_dict.get(first_node_id)
    last_node = nodes_dict.get(last_node_id)
    
    if not first_node or not last_node:
        return None, None
    
    first_point = {"lat": first_node["lat"], "lon": first_node["lon"]}
    last_point = {"lat": last_node["lat"], "lon": last_node["lon"]}
    
    return first_point, last_point


def order_ways_by_connectivity(way_members, ways_dict, nodes_dict):
    """
    Ordena los ways basándose en la conectividad real.
    Construye una cadena continua de ways conectados.
    """
    if not way_members:
        return []
    
    # Crear un diccionario de endpoints para cada way
    way_endpoints = {}
    for member in way_members:
        way_id = member["way_id"]
        way = member["way"]
        first, last = get_way_endpoints(way, nodes_dict)
        if first and last:
            way_endpoints[way_id] = {
                "first": first,
                "last": last,
                "member": member
            }
    
    if not way_endpoints:
        return way_members
    
    # Empezar con el primer way disponible
    ordered = []
    remaining = {wid: data for wid, data in way_endpoints.items()}
    
    # Tomar el primer way
    first_way_id = list(remaining.keys())[0]
    first_data = remaining.pop(first_way_id)
    ordered.append(first_data["member"])
    current_end = first_data["last"]
    
    # Construir la cadena conectando ways
    while remaining:
        best_match = None
        best_distance = float('inf')
        best_reverse = False
        
        for way_id, data in remaining.items():
            # Probar orientación normal
            if points_are_close(current_end, data["first"]):
                distance = 0
                if distance < best_distance:
                    best_match = way_id
                    best_distance = distance
                    best_reverse = False
            # Probar orientación invertida
            elif points_are_close(current_end, data["last"]):
                distance = 0
                if distance < best_distance:
                    best_match = way_id
                    best_distance = distance
                    best_reverse = True
        
        if best_match:
            # Añadir el way que mejor conecta
            data = remaining.pop(best_match)
            if best_reverse:
                # Marcar que este way debe invertirse
                data["member"]["needs_reverse"] = True
            ordered.append(data["member"])
            current_end = data["last"] if not best_reverse else data["first"]
        else:
            # Si no hay conexión, tomar cualquier way restante
            # Esto puede indicar una ruta con múltiples segmentos
            way_id = list(remaining.keys())[0]
            data = remaining.pop(way_id)
            ordered.append(data["member"])
            current_end = data["last"]
    
    return ordered


def osm_relation_to_gpx(relation, nodes_dict, ways_dict, output_path):
    """
    Convierte una relación OSM a GPX respetando el orden correcto y conectando ways correctamente
    
    Args:
        relation: diccionario con la relación OSM
        nodes_dict: diccionario {node_id: node_data}
        ways_dict: diccionario {way_id: way_data}
        output_path: ruta donde guardar el GPX
    """
    gpx = gpxpy.gpx.GPX()
    
    # Añadir metadatos de la ruta
    gpx.name = relation.get("tags", {}).get("name", f"Route {relation['id']}")
    gpx.description = relation.get("tags", {}).get("description", "")
    
    track = gpxpy.gpx.GPXTrack()
    track.name = gpx.name
    gpx.tracks.append(track)
    
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    # Recopilar todos los ways válidos con sus roles
    way_members = []
    for member in relation.get("members", []):
        if member["type"] != "way":
            continue
        
        role = member.get("role", "")
        if role not in ["", "forward", "backward"]:
            continue
        
        way_id = member["ref"]
        if way_id not in ways_dict:
            continue
        
        way_members.append({
            "way_id": way_id,
            "way": ways_dict[way_id],
            "role": role,
            "needs_reverse": False  # Se establecerá durante el ordenamiento
        })
    
    if not way_members:
        # Guardar GPX vacío si no hay ways
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(gpx.to_xml())
        return
    
    # Ordenar ways por conectividad
    ordered_ways = order_ways_by_connectivity(way_members, ways_dict, nodes_dict)
    
    # Procesar cada way en el orden correcto
    last_point = None
    
    for member in ordered_ways:
        way = member["way"]
        role = member["role"]
        needs_reverse = member.get("needs_reverse", False)
        
        # Determinar orientación: prioridad a needs_reverse (conectividad), luego role
        if needs_reverse:
            use_reverse = True
        else:
            # Si no está marcado para invertir, usar el role
            use_reverse = (role == "backward")
        
        way_points = get_way_points(way, nodes_dict, reverse=use_reverse)
        
        # Evitar conexión circular dentro del way
        if len(way_points) > 1:
            if points_are_close(way_points[0], way_points[-1]):
                way_points = way_points[:-1]
        
        # Eliminar el primer punto si es duplicado del último punto ya añadido
        if last_point and len(way_points) > 0:
            if points_are_close(last_point, way_points[0]):
                way_points = way_points[1:]  # Saltar el primer punto duplicado
        
        # Añadir puntos al segmento
        for point in way_points:
            segment.points.append(
                gpxpy.gpx.GPXTrackPoint(
                    latitude=point["lat"],
                    longitude=point["lon"],
                    elevation=None
                )
            )
        
        # Actualizar último punto
        if way_points:
            last_point = way_points[-1]

    # Añadir elevaciones usando datos SRTM
    elevation_data = srtm.get_data()
    elevation_data.add_elevations(gpx)

    # Guardar el archivo GPX
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())
