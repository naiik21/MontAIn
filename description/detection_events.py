
def group_continuous_segments(df, condition_col):
    """
    Agrupa segmentos continuos donde condition_col es True.
    Devuelve lista de (start_km, end_km)
    """
    segments = []
    in_segment = False

    for _, row in df.iterrows():
        if row[condition_col] and not in_segment:
            start_km = row["distance_km"]
            in_segment = True

        elif not row[condition_col] and in_segment:
            end_km = row["distance_km"]
            segments.append((start_km, end_km))
            in_segment = False

    # Si el último tramo llega hasta el final del DataFrame y sigue activo,
    # cerramos el segmento usando la última distancia disponible.
    if in_segment:
        end_km = df.iloc[-1]["distance_km"]
        segments.append((start_km, end_km))

    return segments


def merge_segments(segments, max_gap_km=0.5):
    """
    Une segmentos consecutivos si la separación entre el final de uno
    y el inicio del siguiente es menor que max_gap_km.

    segments: lista de (start_km, end_km) asumida ordenada por start_km.
    """
    if not segments:
        return []

    merged = []
    current_start, current_end = segments[0]

    for start, end in segments[1:]:
        # Si la distancia entre el final actual y el inicio del siguiente
        # es menor que el umbral, los unimos.
        if start - current_end <= max_gap_km:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged

# ==============================
# DETECCIÓN DE EVENTOS
# ==============================

def detect_climbs(df, slope_threshold=35, merge_gap_km=0.5):
    df["strong_climb"] = df["slope"] > slope_threshold
    segments = group_continuous_segments(df, "strong_climb")
    segments = merge_segments(segments, max_gap_km=merge_gap_km)

    return [
        {"type": "climb", "start_km": s, "end_km": e}
        for s, e in segments
    ]

def detect_descents(df, slope_threshold=-35, merge_gap_km=0.5):
    df["strong_descent"] = df["slope"] < slope_threshold
    segments = group_continuous_segments(df, "strong_descent")
    segments = merge_segments(segments, max_gap_km=merge_gap_km)

    return [
        {"type": "descent", "start_km": s, "end_km": e}
        for s, e in segments
    ]

def detect_sharp_turns(df, angle_threshold=359):
    df["turn_angle"] = df["bearing"].diff().abs()

    events = []
    for _, row in df.iterrows():
        if row["turn_angle"] > angle_threshold:
            direction = "derecha" if row["bearing"] > 0 else "izquierda"

            events.append({
                "type": "sharp_turn",
                "km": row["distance_km"],
                "direction": direction
            })

    return events

def detect_technical_sections(df):
    events = []

    for _, row in df.iterrows():
        if row["slope"] > 35 and row.get("rugosity", 0) > 0.6:
            events.append({
                "type": "technical_section",
                "km": row["distance_km"]
            })

    return events



def extract_route_events(df):
    """
    Devuelve lista estructurada de eventos clave
    """
    events = []

    events += detect_climbs(df)
    events += detect_descents(df)
    events += detect_sharp_turns(df)
    events += detect_technical_sections(df)

    return sorted(events, key=lambda x: x.get("start_km", x.get("km", 0)))

