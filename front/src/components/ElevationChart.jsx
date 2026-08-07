import { useEffect, useId, useMemo, useRef, useState } from 'react'

/**
 * Perfil de elevacion con tintas hipsometricas: el relleno toma el color de
 * la altitud, como en un mapa topografico (prado, bosque, ocre, siena, roca,
 * nieve). El color es dato, no decoracion: dos rutas a distinta cota se ven
 * distintas de un vistazo.
 */

// [altitud minima en m, color]
const BANDS = [
  [0, '#5c8a5e'],
  [600, '#7aa365'],
  [1200, '#c2a25c'],
  [1800, '#a97a50'],
  [2400, '#8d8d90'],
  [3000, '#ecece4'],
]

const colorAt = (ele) => {
  let color = BANDS[0][1]
  for (const [from, c] of BANDS) {
    if (ele >= from) color = c
  }
  return color
}

const fmt = (n, digits = 0) =>
  new Intl.NumberFormat('es-ES', { maximumFractionDigits: digits }).format(n)

/** Paso de rejilla que produce entre 3 y 6 marcas. */
function niceStep(range, candidates) {
  for (const step of candidates) {
    if (range / step <= 6) return step
  }
  return candidates[candidates.length - 1]
}

/*
 * El viewBox se adapta al ancho disponible. Con uno fijo de 720 px, en un
 * movil de 375 el SVG se escala a menos de la mitad y las etiquetas de los
 * ejes (10 px) quedan por debajo de 5 px reales, ilegibles. Con un viewBox
 * mas estrecho el escalado es casi 1:1 y el texto se lee.
 */
const DESKTOP = { W: 720, H: 300, PAD: { left: 46, right: 14, top: 18, bottom: 30 } }
const MOBILE = { W: 360, H: 260, PAD: { left: 40, right: 10, top: 16, bottom: 26 } }

function useChartBox() {
  const [box, setBox] = useState(DESKTOP)

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 560px)')
    const apply = () => setBox(mq.matches ? MOBILE : DESKTOP)
    apply()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [])

  return box
}

export function ElevationChart({ elevationData }) {
  const gradientId = useId()
  const svgRef = useRef(null)
  const [hover, setHover] = useState(null)
  const { W, H, PAD } = useChartBox()

  const geo = useMemo(() => {
    const rawX = elevationData?.distance_km ?? []
    const rawY = elevationData?.elevation_m ?? []
    if (rawX.length < 2) return null

    // Puntos sin elevacion (SRTM sin dato) se descartan en pareja.
    let xs = []
    let ys = []
    for (let i = 0; i < rawX.length; i++) {
      if (rawY[i] != null) {
        xs.push(rawX[i])
        ys.push(rawY[i])
      }
    }
    if (xs.length < 2) return null

    // Con miles de puntos el SVG pesa sin ganar nada: se muestrea
    // conservando siempre la cota maxima y el ultimo punto.
    if (xs.length > 800) {
      const stride = Math.ceil(xs.length / 800)
      let maxIdx = 0
      ys.forEach((v, i) => {
        if (v > ys[maxIdx]) maxIdx = i
      })
      const keptX = []
      const keptY = []
      for (let i = 0; i < xs.length; i += stride) {
        keptX.push(xs[i])
        keptY.push(ys[i])
      }
      if ((xs.length - 1) % stride !== 0) {
        keptX.push(xs[xs.length - 1])
        keptY.push(ys[ys.length - 1])
      }
      if (maxIdx % stride !== 0) {
        // se inserta en orden
        const pos = keptX.findIndex((v) => v > xs[maxIdx])
        keptX.splice(pos === -1 ? keptX.length : pos, 0, xs[maxIdx])
        keptY.splice(pos === -1 ? keptY.length : pos, 0, ys[maxIdx])
      }
      xs = keptX
      ys = keptY
    }

    const xMax = xs[xs.length - 1]
    const dataMin = Math.min(...ys)
    const dataMax = Math.max(...ys)

    const yStep = niceStep(dataMax - dataMin || 100, [50, 100, 200, 250, 500, 1000])
    const yMin = Math.floor(dataMin / yStep) * yStep
    const yMax = Math.ceil(dataMax / yStep) * yStep || yStep
    const xStep = niceStep(xMax, [0.5, 1, 2, 5, 10, 20, 50])

    const plotW = W - PAD.left - PAD.right
    const plotH = H - PAD.top - PAD.bottom
    const toX = (km) => PAD.left + (km / xMax) * plotW
    const toY = (m) => PAD.top + ((yMax - m) / (yMax - yMin)) * plotH

    const line = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${toX(x).toFixed(1)} ${toY(ys[i]).toFixed(1)}`).join(' ')
    const area = `${line} L${toX(xMax).toFixed(1)} ${(H - PAD.bottom).toFixed(1)} L${PAD.left} ${(H - PAD.bottom).toFixed(1)} Z`

    // Paradas del degradado: una por frontera de banda dentro del dominio.
    const stops = [{ offset: 0, color: colorAt(yMax) }]
    for (let i = BANDS.length - 1; i >= 0; i--) {
      const boundary = BANDS[i][0]
      if (boundary > yMin && boundary < yMax) {
        stops.push({ offset: (yMax - boundary) / (yMax - yMin), color: BANDS[i][1] })
      }
    }
    stops.push({ offset: 1, color: colorAt(yMin) })

    const yTicks = []
    for (let m = yMin; m <= yMax; m += yStep) yTicks.push(m)
    const xTicks = []
    for (let km = 0; km <= xMax; km += xStep) xTicks.push(km)

    let maxIdx = 0
    ys.forEach((v, i) => {
      if (v > ys[maxIdx]) maxIdx = i
    })

    return { xs, ys, xMax, yMin, yMax, toX, toY, line, area, stops, yTicks, xTicks, maxIdx, xStep }
  }, [elevationData, W, H, PAD])

  if (!geo) return null

  const { xs, ys, xMax, toX, toY, line, area, stops, yTicks, xTicks, maxIdx, xStep } = geo

  const onPointerMove = (e) => {
    const rect = svgRef.current.getBoundingClientRect()
    const km = ((e.clientX - rect.left) * (W / rect.width) - PAD.left) / (W - PAD.left - PAD.right) * xMax
    if (km < 0 || km > xMax) {
      setHover(null)
      return
    }
    // busqueda binaria del punto mas cercano (xs esta ordenado)
    let lo = 0
    let hi = xs.length - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (xs[mid] < km) lo = mid
      else hi = mid
    }
    setHover(km - xs[lo] < xs[hi] - km ? lo : hi)
  }

  const kmDigits = xStep < 1 ? 1 : 0
  const maxLabelX = Math.min(Math.max(toX(xs[maxIdx]), PAD.left + 30), W - PAD.right - 34)

  return (
    <svg
      ref={svgRef}
      className='profile-svg'
      viewBox={`0 0 ${W} ${H}`}
      role='img'
      aria-label={`Perfil de elevación: de ${fmt(geo.yMin)} a ${fmt(ys[maxIdx])} metros a lo largo de ${fmt(xMax, 1)} kilómetros`}
      onPointerMove={onPointerMove}
      onPointerLeave={() => setHover(null)}
    >
      <defs>
        <linearGradient id={gradientId} x1='0' y1={PAD.top} x2='0' y2={H - PAD.bottom} gradientUnits='userSpaceOnUse'>
          {stops.map((s, i) => (
            <stop key={i} offset={s.offset} stopColor={s.color} />
          ))}
        </linearGradient>
      </defs>

      {/* rejilla y eje y (cotas) */}
      {yTicks.map((m) => (
        <g key={m}>
          <line className='profile-grid-line' x1={PAD.left} x2={W - PAD.right} y1={toY(m)} y2={toY(m)} />
          <text className='profile-axis-label' x={PAD.left - 7} y={toY(m) + 3.5} textAnchor='end'>
            {fmt(m)}
          </text>
        </g>
      ))}

      {/* eje x (kilometros) */}
      {xTicks.map((km, i) => (
        <text
          key={km}
          className='profile-axis-label'
          x={toX(km)}
          y={H - PAD.bottom + 16}
          textAnchor={i === 0 ? 'start' : 'middle'}
        >
          {fmt(km, kmDigits)}
          {i === xTicks.length - 1 ? ' km' : ''}
        </text>
      ))}
      <text className='profile-axis-label' x={PAD.left - 7} y={PAD.top - 7} textAnchor='end'>
        m
      </text>

      {/* la ruta: relleno hipsometrico + trazo de tinta */}
      <path d={area} fill={`url(#${gradientId})`} fillOpacity='0.88' />
      <path d={line} className='profile-line' />

      {/* cota maxima */}
      <circle cx={toX(xs[maxIdx])} cy={toY(ys[maxIdx])} r='3' fill='var(--ink)' />
      <text className='profile-peak-label' x={maxLabelX} y={toY(ys[maxIdx]) - 8} textAnchor='middle'>
        {fmt(ys[maxIdx])} m
      </text>

      {/* lectura bajo el cursor */}
      {hover != null && (
        <g>
          <line
            className='profile-cursor-line'
            x1={toX(xs[hover])}
            x2={toX(xs[hover])}
            y1={PAD.top}
            y2={H - PAD.bottom}
          />
          <circle cx={toX(xs[hover])} cy={toY(ys[hover])} r='3.5' fill='var(--ink)' />
          {(() => {
            const tipW = 82
            const flip = toX(xs[hover]) > W - tipW - 20
            const bx = flip ? toX(xs[hover]) - tipW - 10 : toX(xs[hover]) + 10
            const by = Math.max(PAD.top + 2, toY(ys[hover]) - 40)
            return (
              <g>
                <rect className='profile-tip-box' x={bx} y={by} width={tipW} height='34' rx='4' />
                <text className='profile-tip-text' x={bx + 10} y={by + 14}>
                  {fmt(xs[hover], 1)} km
                </text>
                <text className='profile-tip-text' x={bx + 10} y={by + 27}>
                  {fmt(ys[hover])} m
                </text>
              </g>
            )
          })()}
        </g>
      )}
    </svg>
  )
}
