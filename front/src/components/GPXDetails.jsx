import '../style/index.css'

/**
 * Componente para mostrar los detalles del GPX procesado
 */
export function GPXDetails({ gpxData }) {

  if (!gpxData) {
    return (
      <div className="section">
        <div className="section-title">Detalles del GPX</div>
        <div style={{ fontSize: '14px', color: '#444444' }}>
          Aún no se ha procesado ninguna ruta.
        </div>
      </div>
    )
  }

  console.log(gpxData.difficulty)

  return (
    <div className="section">
      <div className="section-title">Detalles del GPX</div>
      <div style={{ fontSize: '14px', color: '#444444' }}>
        <p>
          <strong>Nombre GPX interno:</strong> {gpxData.filename || 'N/A'}
        </p>
        <p>
          <strong>Distance:</strong> {gpxData.distance_km ?? 'N/A'} km
        </p>
        <p>
          <strong>Elevation Gain:</strong> {gpxData.elevation_gain ?? 'N/A'} m
        </p>
        <p>
          <strong>Elevation Loss:</strong> {gpxData.elevation_loss ?? 'N/A'} m
        </p>
        <p>
          <strong>Max Elevation:</strong> {gpxData.max_elevation ?? 'N/A'} m
        </p>
        <p>
          <strong>Min Elevation:</strong> {gpxData.min_elevation ?? 'N/A'} m
        </p>
        <p>
          <strong>Max Slope:</strong> {gpxData.max_slope ?? 'N/A'}°
        </p>
        <p>
          <strong>Mean Slope:</strong> {gpxData.mean_slope ?? 'N/A'}°
        </p>
        <p>
          <strong>% Over 30°:</strong>{' '}
          {gpxData.pct_over_30 != null
            ? (gpxData.pct_over_30 * 100).toFixed(1) + '%'
            : 'N/A'}
        </p>
        <p>
          <strong>% Over 40°:</strong>{' '}
          {gpxData.pct_over_40 != null
            ? (gpxData.pct_over_40 * 100).toFixed(1) + '%'
            : 'N/A'}
        </p>
        <p>
          <strong>% Over 45°:</strong>{' '}
          {gpxData.pct_over_45 != null
            ? (gpxData.pct_over_45 * 100).toFixed(1) + '%'
            : 'N/A'}
        </p>
        <p>
          <strong>Mean Aspect:</strong> {gpxData.mean_aspect ?? 'N/A'}°
        </p>
        <p>
          <strong>Rugosity Mean:</strong> {gpxData.rugosity_mean ?? 'N/A'}
        </p>
        <p>
          <strong>Exposed %:</strong>{' '}
          {gpxData.exposed_pct != null
            ? (gpxData.exposed_pct * 100).toFixed(1) + '%'
            : 'N/A'}
        </p>
        <p>
          <strong>Difficulty:</strong> {gpxData.difficulty || 'N/A'}
        </p>
      </div>
    </div>
  )
}

