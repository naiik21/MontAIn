def generate_route_guide(route: dict, difficulty_label: str) -> dict:
    guide = {}

    guide["difficulty"] = difficulty_label[0]

    # --- Esfuerzo ---
    if route["distance_km"] > 20 or route["elevation_gain"] > 1500:
        guide["effort_level"] = "muy alto"
    elif route["distance_km"] > 12:
        guide["effort_level"] = "alto"
    else:
        guide["effort_level"] = "moderado"

    # --- Técnica ---
    guide["requires_hands"] = (
        route["pct_over_45"] > 0.01 or route["max_slope"] > 42
    )

    guide["has_exposure"] = (
        route["exposed_pct"] > 0.05  or
      route["pct_over_40"] > 0.15
    )

    # --- Nivel técnico ---
    if route["max_slope"] > 45:
        guide["technical_level"] = "muy alto"
    elif route["max_slope"] > 35:
        guide["technical_level"] = "alto"
    else:
        guide["technical_level"] = "medio"

    # --- Terreno ---
    terrain = []
    if route["rugosity_mean"] > 1.5:
        terrain.append("terreno abrupto")
    if route["mean_slope"] > 20:
        terrain.append("pendientes mantenidas")
    guide["terrain_description"] = terrain

    # --- Riesgos ---
    risks = []
    if guide["has_exposure"]:
        risks.append("exposición")
    if route["distance_km"] > 18:
        risks.append("fatiga acumulada")
    if route["max_elevation"] > 2500:
        risks.append("cambios meteorológicos")
    guide["main_risks"] = risks

    # --- Perfil recomendado ---
    if guide["technical_level"] in ["alto", "muy alto"]:
        guide["recommended_profile"] = "personas con experiencia en montaña"
    else:
        guide["recommended_profile"] = "montañeros habituados a rutas exigentes"

    # --- Consejos ---
    advice = []
    if guide["has_exposure"]:
        advice.append("evitar la ruta con viento fuerte o terreno mojado")
    if guide["effort_level"] == "muy alto":
        advice.append("salir temprano y gestionar bien los tiempos")
    guide["general_advice"] = advice

    return guide

def render_guide_text(guide: dict) -> str:
    paragraphs = []

    # --- Resumen ---
    summary = (
        f"Ruta de {guide['difficulty']} con un nivel de esfuerzo "
        f"{guide['effort_level']} y un nivel técnico {guide['technical_level']}."
    )
    paragraphs.append(summary)

    # --- Terreno ---
    if guide["terrain_description"]:
        terrain = ", ".join(guide["terrain_description"])
        paragraphs.append(
            f"El recorrido discurre por {terrain}, lo que requiere atención "
            f"y una progresión segura."
        )

    # --- Técnica ---
    if guide["requires_hands"]:
        paragraphs.append(
            "Existen tramos donde puede ser necesario el uso puntual de las manos "
            "para progresar con seguridad."
        )
    else:
        paragraphs.append(
            "No se requieren técnicas de escalada ni el uso de las manos."
        )

    # --- Exposición ---
    if guide["has_exposure"]:
        paragraphs.append(
            "Algunos pasos son expuestos, por lo que se recomienda extremar la "
            "precaución, especialmente con viento o terreno mojado."
        )

    # --- Riesgos ---
    if guide["main_risks"]:
        risks = ", ".join(guide["main_risks"])
        paragraphs.append(
            f"Los principales riesgos de la ruta son: {risks}."
        )

    # --- Perfil recomendado ---
    paragraphs.append(
        f"Ruta recomendada para {guide['recommended_profile']}."
    )

    # --- Consejos ---
    if guide["general_advice"]:
        advice = " ".join(guide["general_advice"])
        paragraphs.append(
            f"Recomendaciones: {advice}."
        )

    return "\n\n".join(paragraphs)
