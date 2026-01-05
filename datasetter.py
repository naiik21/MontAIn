import numpy as np
from geopy.distance import geodesic
import pandas as pd
import glob
from GPX_uses.gpx_loader import load_gpx


def compute_distances(df):
    """
    Calcula la distancia horizontal entre puntos consecutivos del track.
    
    Usa cálculo geodesic preciso para medir la distancia en metros sobre la superficie
    de la Tierra entre cada par de puntos consecutivos del track GPS.
    El primer punto tiene distancia 0 ya que no hay punto previo.
    """
    distances = [0.0]

    for i in range(1, len(df)):
        p1 = (df.loc[i-1, "lat"], df.loc[i-1, "lon"])
        p2 = (df.loc[i, "lat"], df.loc[i, "lon"])
        dist = geodesic(p1, p2).meters
        distances.append(dist)

    df["dist_segment"] = distances
    return df


def compute_slope(df):
    """
    Calcula la pendiente (slope) en grados entre puntos consecutivos del track.
    
    La pendiente es el ángulo de inclinación del terreno entre dos puntos consecutivos,
    calculado como el arco tangente del cociente entre la diferencia de elevación 
    y la distancia horizontal: arctan(diferencia_elevación / distancia_horizontal).
    
    El resultado está en GRADOS (ángulo), NO en porcentaje:
    - 0° = terreno completamente plano
    - 5-10° = ligera pendiente (caminata fácil)
    - 15-20° = pendiente moderada
    - 30-35° = pendiente pronunciada (escalada fácil)
    - 45° = muy empinado (escalada moderada)
    - 60-70° = extremadamente empinado (escalada difícil)
    - 90° = vertical (escalada técnica)
    
    Valores positivos = subida, negativos = bajada.
    Rango: -90° a +90° (aunque en la práctica raramente supera ±45° en senderos).
    """
    slopes = [0.0] 

    for i in range(1, len(df)):
        p1 = (df.loc[i-1, "lat"], df.loc[i-1, "lon"])
        p2 = (df.loc[i, "lat"], df.loc[i, "lon"])

        # Calcula distancia horizontal usando geodesic (más preciso para coordenadas geográficas)
        dist = geodesic(p1, p2).meters
        elev_diff = df.loc[i, "ele"] - df.loc[i-1, "ele"]

        # Evita división por cero y errores con distancias muy pequeñas
        # También verifica que las elevaciones no sean NaN
        if dist < 0.1 or pd.isna(elev_diff) or pd.isna(dist):
            slopes.append(0.0)
            continue

        # Calcula pendiente en grados: arctan(elevación/distancia)
        slope_rad = np.arctan(elev_diff / dist)
        slopes.append(np.degrees(slope_rad))

    df["slope_deg"] = slopes
    return df


def compute_aspect(df):
    """
    Calcula el acimut (aspect/bearing) de la dirección de viaje entre puntos consecutivos.
    
    El aspect indica la dirección cardinal hacia donde te diriges:
    - 0° o 360° = Norte
    - 90° = Este  
    - 180° = Sur
    - 270° = Oeste
    
    Usa fórmulas trigonométricas precisas para calcular el bearing sobre una esfera,
    considerando la curvatura de la Tierra.
    """
    aspects = [0.0]  # Primer punto no tiene dirección previa

    for i in range(1, len(df)):
        lat1 = df.loc[i-1, "lat"]
        lon1 = df.loc[i-1, "lon"]
        lat2 = df.loc[i, "lat"]
        lon2 = df.loc[i, "lon"]

        # Verifica valores válidos
        if pd.isna(lat1) or pd.isna(lat2) or pd.isna(lon1) or pd.isna(lon2):
            aspects.append(0.0)
            continue

        # Convierte a radianes
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)

        # Fórmula para calcular el bearing inicial usando la fórmula del acimut
        # Más precisa que diferencias simples de lat/lon
        x = np.sin(dlon_rad) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
        
        # Calcula el ángulo y normaliza a 0-360°
        bearing_rad = np.arctan2(x, y)
        bearing_deg = np.degrees(bearing_rad)
        aspect = (bearing_deg + 360) % 360
        
        aspects.append(aspect)

    df["aspect_deg"] = aspects
    return df


def compute_exposure(df, slope_threshold=35, window=5):
    """
    Identifica secciones expuestas (exposure) basándose en pendientes pronunciadas.
    
    Una sección se considera "expuesta" si tiene pendientes superiores al umbral
    (por defecto 35°) durante una ventana de puntos consecutivos.
    
    Esto es útil para identificar terrenos peligrosos donde un resbalón
    podría tener consecuencias graves (ej: bordes de acantilados, crestas estrechas).
    
    Parámetros:
    - slope_threshold: Pendiente mínima en grados para considerar "crítico" (default: 35°)
    - window: Número de puntos consecutivos que deben cumplir el umbral (default: 5)
    """
    # Identifica puntos con pendiente crítica
    critical = df["slope_deg"].abs() > slope_threshold  # Usa valor absoluto para subidas y bajadas
    
    # Calcula ventana móvil: una sección es expuesta si todos los puntos en la ventana son críticos
    # .sum() >= window significa que todos los puntos en la ventana cumplen la condición
    exposed_rolling = critical.rolling(window=window, min_periods=1).sum() >= window
    
    # Llena los primeros valores (donde no hay suficientes puntos para la ventana)
    # con False ya que no podemos determinar exposición sin suficientes datos
    df["exposed"] = exposed_rolling.fillna(False).astype(bool)
    
    return df


def compute_rugosity(df, window=5):
    """
    Calcula la rugosidad del terreno (rugosity/roughness).
    
    La rugosidad mide qué tan variable y "accidentado" es el terreno. 
    Un valor alto indica terreno con muchas variaciones (pendientes que cambian 
    rápidamente, terreno irregular). Un valor bajo indica terreno suave y uniforme.
    
    La medida combina:
    1. Cambios rápidos en pendiente (variabilidad de la inclinación)
    2. Cambios relativos en elevación normalizados por distancia
    
    Parámetros:
    - window: Tamaño de la ventana móvil para suavizar el cálculo (default: 5)
    """
    # Calcula la variabilidad de pendiente (cambio absoluto entre puntos consecutivos)
    slope_variability = df["slope_deg"].diff().abs()
    
    # Calcula cambio relativo de elevación normalizado por distancia
    # Esto captura variaciones bruscas de elevación
    elev_change = df["ele"].diff().abs()
    dist_segment = df["dist_segment"].replace(0, np.nan)  # Evita división por cero
    
    # Normaliza cambios de elevación por distancia (metros de cambio por metro recorrido)
    elev_roughness = elev_change / dist_segment
    
    # Combina ambas medidas (variabilidad de pendiente y cambios bruscos de elevación)
    # Peso mayor a variabilidad de pendiente ya que es más estable
    combined_roughness = slope_variability * 0.7 + elev_roughness.fillna(0) * 0.3
    
    # Suaviza con ventana móvil para evitar valores extremos aislados
    df["rugosity"] = combined_roughness.rolling(window=window, min_periods=1).mean()
    
    # Llena NaN (primeros valores sin ventana completa) con 0
    df["rugosity"] = df["rugosity"].fillna(0)
    
    return df


def enrich_features(df):
    df = compute_distances(df)
    df = compute_slope(df)
    df = compute_aspect(df)
    df = compute_exposure(df)
    df = compute_rugosity(df)
    return df


def aggregate_route_features(df):
    """
    Agrega características estadísticas de una ruta completa.
    
    Calcula métricas agregadas del track GPS incluyendo distancias, elevaciones,
    pendientes, direcciones y características del terreno.
    """
    # Distancia total en kilómetros
    distance_km = df["dist_segment"].sum() / 1000
    
    # Ganancia y pérdida de elevación (solo valores positivos)
    elev_diffs = df["ele"].diff()
    elevation_gain = elev_diffs.clip(lower=0).sum()  # Solo subidas
    elevation_loss = (-elev_diffs.clip(upper=0)).sum()  # Solo bajadas (convertidas a positivo)
    
    # Elevación máxima y mínima
    max_elevation = df["ele"].max() if len(df["ele"].dropna()) > 0 else 0.0
    min_elevation = df["ele"].min() if len(df["ele"].dropna()) > 0 else 0.0
    
    # Pendientes: manejo de NaN y cálculo de estadísticas
    slopes_abs = df["slope_deg"].abs()  # Pendiente absoluta (subidas y bajadas)
    slopes_valid_abs = slopes_abs.dropna()
    
    max_slope = slopes_abs.max() if len(slopes_valid_abs) > 0 else 0.0
    # mean_slope: pendiente promedio (valor absoluto) - qué tan empinada es la ruta en general
    mean_slope = slopes_valid_abs.mean() if len(slopes_valid_abs) > 0 else 0.0
    
    # Porcentajes de pendientes empinadas (considerando subidas y bajadas)
    total_points = len(slopes_valid_abs)
    if total_points > 0:
        pct_over_30 = (slopes_valid_abs > 30).sum() / total_points
        pct_over_40 = (slopes_valid_abs > 40).sum() / total_points
        pct_over_45 = (slopes_valid_abs > 45).sum() / total_points
    else:
        pct_over_30 = pct_over_40 = pct_over_45 = 0.0
    
    # Media circular para aspect (ángulos 0-360°)
    # La media aritmética simple no funciona para ángulos circulares
    aspects_valid = df["aspect_deg"].dropna()
    if len(aspects_valid) > 0:
        # Convierte a radianes, calcula seno y coseno, y luego la media circular
        aspects_rad = np.radians(aspects_valid)
        mean_sin = np.sin(aspects_rad).mean()
        mean_cos = np.cos(aspects_rad).mean()
        mean_aspect_rad = np.arctan2(mean_sin, mean_cos)
        mean_aspect = (np.degrees(mean_aspect_rad) + 360) % 360
    else:
        mean_aspect = 0.0
    
    # Rugosidad y exposición
    rugosity_mean = df["rugosity"].mean() if len(df["rugosity"].dropna()) > 0 else 0.0
    exposed_pct = df["exposed"].mean() if "exposed" in df.columns else 0.0
    
    
    return {
        "distance_km": round(distance_km, 2),
        "elevation_gain": round(elevation_gain, 2),
        "elevation_loss": round(elevation_loss, 2),
        "max_elevation": round(max_elevation, 2),
        "min_elevation": round(min_elevation, 2),
        "max_slope": round(max_slope, 2),
        "mean_slope": round(mean_slope, 2),
        "pct_over_30": round(pct_over_30, 2),
        "pct_over_40": round(pct_over_40, 2),
        "pct_over_45": round(pct_over_45, 2),
        "mean_aspect": round(mean_aspect, 2),
        "rugosity_mean": round(rugosity_mean, 2),
        "exposed_pct": round(exposed_pct, 2)
    }


def get_difficulty_name(difficulty_level):
    """
    Obtiene el nombre descriptivo del nivel de dificultad.
    
    Parámetros:
    - difficulty_level: int (0-5)
    
    Retorna:
    - str: nombre de la dificultad
    """
    difficulty_names = {
        0: "sendero fácil",
        1: "moderado",
        2: "difícil",
        3: "alta montaña",
        4: "alpinismo ligero",
        5: "alpinismo técnico"
    }
    return difficulty_names.get(difficulty_level, "desconocido")


def classify_difficulty(features):
    """
    Clasifica la dificultad de una ruta basándose en variables objetivas.
    
    Niveles de dificultad:
    0 = sendero fácil
    1 = moderado
    2 = difícil
    3 = alta montaña
    4 = alpinismo ligero
    5 = alpinismo técnico
    
    Parámetros:
    - features: diccionario con las características de la ruta
    
    Retorna:
    - int: nivel de dificultad (0-5)
    """
    max_slope = features.get("max_slope", 0)
    mean_slope = features.get("mean_slope", 0)
    elevation_gain = features.get("elevation_gain", 0)
    max_elevation = features.get("max_elevation", 0)
    pct_over_30 = features.get("pct_over_30", 0)
    pct_over_40 = features.get("pct_over_40", 0)
    pct_over_45 = features.get("pct_over_45", 0)
    exposed_pct = features.get("exposed_pct", 0)
    rugosity_mean = features.get("rugosity_mean", 0)
    
    # Inicializar puntuación de dificultad
    difficulty_score = 0
    
    # Factor 1: Pendiente máxima (más importante para alpinismo técnico)
    if max_slope >= 80:  # Casi vertical
        difficulty_score += 3
    elif max_slope >= 70:  # Muy empinado
        difficulty_score += 2
    elif max_slope >= 60:  # Empinado
        difficulty_score += 1
    elif max_slope >= 50:  # Moderadamente empinado
        difficulty_score += 0.5
    
    # Factor 2: Pendiente promedio
    if mean_slope >= 15:  # Muy empinado en promedio
        difficulty_score += 2
    elif mean_slope >= 10:  # Empinado en promedio
        difficulty_score += 1
    elif mean_slope >= 5:  # Moderado
        difficulty_score += 0.5
    
    # Factor 3: Porcentaje de pendientes extremas
    if pct_over_45 >= 0.15:  # Más del 15% con pendientes >45°
        difficulty_score += 2
    elif pct_over_45 >= 0.08:  # Más del 8% con pendientes >45°
        difficulty_score += 1
    elif pct_over_40 >= 0.15:  # Más del 15% con pendientes >40°
        difficulty_score += 0.5
    
    if pct_over_30 >= 0.25:  # Más del 25% con pendientes >30°
        difficulty_score += 0.5
    
    # Factor 4: Elevación máxima (alta montaña)
    if max_elevation >= 3000:  # Alta montaña (3000m+)
        difficulty_score += 1.5
    elif max_elevation >= 2500:  # Montaña alta (2500m+)
        difficulty_score += 1
    elif max_elevation >= 2000:  # Montaña media (2000m+)
        difficulty_score += 0.5
    
    # Factor 5: Ganancia de elevación total
    if elevation_gain >= 2000:  # Muy alta ganancia
        difficulty_score += 1
    elif elevation_gain >= 1000:  # Alta ganancia
        difficulty_score += 0.5
    
    # Factor 6: Exposición (terreno peligroso)
    if exposed_pct >= 0.15:  # Más del 15% expuesto
        difficulty_score += 1.5
    elif exposed_pct >= 0.08:  # Más del 8% expuesto
        difficulty_score += 1
    elif exposed_pct >= 0.03:  # Más del 3% expuesto
        difficulty_score += 0.5
    
    # Factor 7: Rugosidad del terreno
    if rugosity_mean >= 15:  # Muy rugoso
        difficulty_score += 1
    elif rugosity_mean >= 10:  # Rugoso
        difficulty_score += 0.5
    
    # Clasificación final basada en puntuación acumulada
    if difficulty_score >= 8:
        return 5  # Alpinismo técnico
    elif difficulty_score >= 6:
        return 4  # Alpinismo ligero
    elif difficulty_score >= 4.5:
        return 3  # Alta montaña
    elif difficulty_score >= 2.5:
        return 2  # Difícil
    elif difficulty_score >= 1:
        return 1  # Moderado
    else:
        return 0  # Sendero fácil


def build_dataset(gpx_dir="data/gpx"):
    rows = []

    for path in glob.glob(f"{gpx_dir}/*.gpx" ):
        df, name = load_gpx(path)
        df = enrich_features(df)
        row = {}
        row["filename"] = name
        features = aggregate_route_features(df)
        row.update(features)
        # Agregar clasificación de dificultad
        row["difficulty"] = get_difficulty_name(classify_difficulty(features))
        rows.append(row)
    print(rows)
        
    return pd.DataFrame(rows)
