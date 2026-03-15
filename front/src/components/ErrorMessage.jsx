import '../style/index.css'

/**
 * Componente para mostrar mensajes de error
 */
export function ErrorMessage({ error }) {
  if (!error) return null

  return <div className="error active">{error}</div>
}

