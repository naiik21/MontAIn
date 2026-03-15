
import datasetter

import numpy as np
from GPX_uses.gpx_loader import gpx_to_dataframe
import description.route_guide as route_guide
import description.build_compute as build_compute
import description.detection_events as detection_events
import description.key_moments as key_moment
import description.anthropic_claude as anthropic_claude
import description.hugging_face as hugging_face
import description.groq as groq





def main():  
    gpx="data/serra-del-cadi.gpx" 
    
    dataset = datasetter.build_dataset_from_file(gpx)
    dataset = dataset.to_dict()

    route={
   	 "distance_km":dataset["distance_km"][0],	"elevation_gain":dataset["elevation_gain"][0],	"elevation_loss":dataset["elevation_loss"][0],	"max_elevation":dataset["max_elevation"][0],	"min_elevation":dataset["min_elevation"][0],	"max_slope":dataset["max_slope"][0],	"mean_slope":dataset["mean_slope"][0],	"pct_over_30":dataset["pct_over_30"][0],	"pct_over_40":dataset["pct_over_40"][0],	"pct_over_45":dataset["pct_over_45"][0],	"mean_aspect":dataset["mean_aspect"][0],	"rugosity_mean":dataset["rugosity_mean"][0],	"exposed_pct":dataset["exposed_pct"][0]
     }
 
    guide = route_guide.generate_route_guide(route, dataset["difficulty"])
    
    # render_guide=route_guide.render_guide_text(guide)
    # print(render_guide)
    
    df = gpx_to_dataframe(gpx)
    df= build_compute.build_features(df)
    # print("-------------")
    # print(detection_events.detect_climbs(df))
    # print("-------------")
    # print(detection_events.detect_descents(df))
    # print("-------------")
    # print(detection_events.detect_sharp_turns(df))
    # print("-------------")
    # print(detection_events.detect_technical_sections(df))

    events= detection_events.extract_route_events(df)
    

    key_moment.compute_fatigue_score(df)
  
    key_moments= key_moment.detect_key_moments(df)
    
    prompt=f"""
        Eres un guía de montaña profesional. 
        Redacta una descripción clara, responsable y realista de una ruta de montaña usando ÚNICAMENTE la información del JSON proporcionado. 

        Normas: 
        - No inventes información. 
        - No añadas pasos técnicos que no estén indicados. 
        - No menciones escalada salvo que se indique explícitamente. 
        - Usa un tono informativo, no épico. - Prioriza la seguridad y la claridad. 
        - No hagas recomendaciones médicas ni técnicas avanzadas. 
 
        JSON: 
        {guide}

        Eventos detectados:
        {events}

        Momentos clave:
        {key_moments}
        """
    # description= anthropic_claude.generate_description(prompt)
    # description= hugging_face.generate_description(prompt, context)
    description= groq.generate_description(prompt)
    print(description)


if __name__ == "__main__":
    main()

 