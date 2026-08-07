import { useEffect, useState } from 'react'
import { Waymark } from './Waymark.jsx'

/*
 * Etapas orientativas, en el orden real del pipeline del backend.
 * No hay streaming de progreso, asi que los tiempos son aproximados;
 * la ultima etapa (la descripcion con IA) es la que domina.
 */
const STAGES = [
  { at: 0, text: 'Leyendo la traza…' },
  { at: 1500, text: 'Consultando elevaciones…' },
  { at: 3000, text: 'Calculando pendientes y exposición…' },
  { at: 4800, text: 'Estimando la dificultad…' },
  { at: 7000, text: 'Redactando la descripción de guía…' },
]

export function LoadingIndicator() {
  const [stage, setStage] = useState(STAGES[0].text)

  useEffect(() => {
    const timers = STAGES.slice(1).map(({ at, text }) =>
      setTimeout(() => setStage(text), at)
    )
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <div className='loading' role='status' aria-live='polite'>
      <Waymark />
      <p className='loading-stage'>{stage}</p>
      <p className='loading-note'>Suele tardar entre 10 y 20 segundos.</p>
    </div>
  )
}
