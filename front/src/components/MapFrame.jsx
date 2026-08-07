/**
 * El HTML del mapa lo genera folium en el servidor. Se aisla en un iframe
 * con sandbox en lugar de inyectarlo con dangerouslySetInnerHTML: sus
 * scripts (Leaflet) corren sin acceso al origen de la app y su CSS no se
 * mezcla con el nuestro.
 */
export function MapFrame({ mapHtml }) {
  if (!mapHtml) return null

  return (
    <iframe
      className='map-frame'
      srcDoc={mapHtml}
      sandbox='allow-scripts'
      title='Mapa del recorrido'
      loading='lazy'
    />
  )
}
