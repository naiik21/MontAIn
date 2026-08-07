const fmt = (n, digits = 0) =>
  new Intl.NumberFormat('es-ES', {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  }).format(n)

const pct = (n) => (n == null ? null : `${fmt(n * 100, 1)}`)

/** 205° -> "SO": la orientacion media como se leeria en una brujula. */
function aspectToCardinal(deg) {
  if (deg == null) return null
  const dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSO', 'SO', 'OSO', 'O', 'ONO', 'NO', 'NNO']
  return dirs[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16]
}

export function MetricsGrid({ record }) {
  if (!record) return null

  const metrics = [
    { label: 'Distancia', value: fmt(record.distance_km, 1), unit: 'km' },
    { label: 'Desnivel +', value: fmt(record.elevation_gain), unit: 'm' },
    { label: 'Desnivel −', value: fmt(record.elevation_loss), unit: 'm' },
    { label: 'Cota máxima', value: fmt(record.max_elevation), unit: 'm' },
    { label: 'Cota mínima', value: fmt(record.min_elevation), unit: 'm' },
    { label: 'Pendiente media', value: fmt(record.mean_slope, 1), unit: '°' },
    { label: 'Pendiente máxima', value: fmt(record.max_slope, 1), unit: '°' },
    { label: 'Tramo > 30°', value: pct(record.pct_over_30), unit: '%' },
    { label: 'Tramo > 40°', value: pct(record.pct_over_40), unit: '%' },
    { label: 'Tramo > 45°', value: pct(record.pct_over_45), unit: '%' },
    {
      label: 'Orientación media',
      value: aspectToCardinal(record.mean_aspect),
      unit: record.mean_aspect != null ? `${fmt(record.mean_aspect)}°` : '',
    },
    { label: 'Rugosidad', value: fmt(record.rugosity_mean, 1), unit: '' },
    { label: 'Terreno expuesto', value: pct(record.exposed_pct), unit: '%' },
  ]

  return (
    <div className='metrics-grid'>
      {metrics
        .filter((m) => m.value != null)
        .map((m) => (
          <div className='metric' key={m.label}>
            <span className='metric-label'>{m.label}</span>
            <span className='metric-value'>
              {m.value}
              {m.unit ? <span className='metric-unit'> {m.unit}</span> : null}
            </span>
          </div>
        ))}
    </div>
  )
}
