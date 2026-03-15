export function ElevationChart({ elevationData }) {
  if (!elevationData) return null

  const distances = elevationData.distance_km || []
  const elevations = elevationData.elevation_m || []

  if (!distances.length || !elevations.length) return null

  const width = 800
  const height = 300

  const maxDistance = Math.max(...distances)
  const minElevation = Math.min(...elevations)
  const maxElevation = Math.max(...elevations)
  const elevationRange = maxElevation - minElevation || 1

  const points = distances.map((d, i) => {
    const x = (d / maxDistance) * width
    const y =
      height - ((elevations[i] - minElevation) / elevationRange) * height
    return { x, y }
  })

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
    .join(' ')

  return (
    <div className="section">
      <div className="section-title">Perfil de elevación</div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height="250"
        preserveAspectRatio="none"
        style={{
          background: '#f5f5f5',
          borderRadius: '8px',
          border: '1px solid #ddd',
        }}
      >
        {/* Eje base */}
        <line
          x1="0"
          y1={height}
          x2={width}
          y2={height}
          stroke="#ccc"
          strokeWidth="1"
        />

        {/* Curva de elevación */}
        <path
          d={pathD}
          fill="none"
          stroke="#007bff"
          strokeWidth="2"
        />

        {/* Relleno bajo la curva */}
        <path
          d={`${pathD} L ${width} ${height} L 0 ${height} Z`}
          fill="rgba(0, 123, 255, 0.1)"
        />
      </svg>
      <div style={{ fontSize: '12px', color: '#555', marginTop: '4px' }}>
        Distancia (km) vs. Elevación (m)
      </div>
    </div>
  )
}


