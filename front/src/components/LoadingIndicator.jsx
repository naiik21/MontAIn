import '../style/index.css'

/**
 * Componente para mostrar el indicador de carga
 */
export function LoadingIndicator({ isLoading }) {
  if (!isLoading) return null

  return (
    <div className="loading active">
      <p>Procesando ruta... Esto puede tardar unos segundos.</p>
    </div>
  )
}

