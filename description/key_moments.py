def compute_fatigue_score(df):
    """
    Modelo simple de fatiga acumulada
    """
    fatigue = 0
    fatigue_profile = []

    for _, row in df.iterrows():
        slope_factor = max(row["slope"], 0)
        fatigue += slope_factor * 0.01
        fatigue_profile.append(fatigue)

    df["fatigue_score"] = fatigue_profile
    return df

def detect_key_moments(df):
    """
    Detecta puntos críticos globales
    """
    max_slope_row = df.loc[df["slope"].idxmax()]
    max_fatigue_row = df.loc[df["fatigue_score"].idxmax()]

    return [
        {
            "type": "max_slope_point",
            "km": max_slope_row["distance_km"]
        },
        {
            "type": "max_fatigue_point",
            "km": max_fatigue_row["distance_km"]
        }
    ]
