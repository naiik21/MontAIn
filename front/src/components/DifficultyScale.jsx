/**
 * Escalera de dificultad: seis grados ordinales con el codigo de color de
 * los deportes de montana (verde facil -> negro experto, como las pistas de
 * esqui). El ultimo grado usa la tinta del tema para seguir siendo visible
 * en modo oscuro.
 */
const LEVELS = [
  { name: 'sendero fácil', color: '#4c8f55' },
  { name: 'moderado', color: '#c9a227' },
  { name: 'difícil', color: '#d07a2c' },
  { name: 'alta montaña', color: '#c13a2a' },
  { name: 'alpinismo ligero', color: '#8c2334' },
  { name: 'alpinismo técnico', color: 'var(--ink)' },
]

// Los nombres llegan del backend tal como los define datasetter.py
// (minusculas y con acentos), asi que basta con una comparacion directa.
const normalize = (s) => (s || '').trim().toLowerCase()

export function DifficultyScale({ difficulty }) {
  const index = LEVELS.findIndex((l) => normalize(l.name) === normalize(difficulty))
  if (index === -1) {
    // Grado desconocido: se muestra el texto tal cual, sin escalera.
    return <span className='grade-label'>{difficulty}</span>
  }

  const level = LEVELS[index]

  return (
    <span
      className='grade'
      role='img'
      aria-label={`Dificultad: ${level.name}, grado ${index + 1} de ${LEVELS.length}`}
    >
      <span className='grade-steps' aria-hidden='true'>
        {LEVELS.map((l, i) => (
          <span
            key={l.name}
            className={`grade-step ${i <= index ? 'grade-step--filled' : ''}`}
            style={i <= index ? { background: LEVELS[i].color } : undefined}
          />
        ))}
      </span>
      <span className='grade-label' style={{ color: level.color }}>
        {level.name}
      </span>
    </span>
  )
}
