import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_hiking_routes(bbox, limit=1000):
    """
    bbox = (south, west, north, east)
    """
    query = f"""
    [out:json][timeout:25];
    (
      relation["route"="hiking"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation["route"="climbing"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body {limit};
    >;
    out skel qt;
    """
    
    """out geom;"""

    response = requests.post(OVERPASS_URL, data=query)
    response.raise_for_status()

    return response.json()
  
  
