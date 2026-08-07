export function ErrorMessage({ error }) {
  if (!error) return null

  return (
    <div className='error' role='alert'>
      <div>
        <p className='error-title'>No se ha podido analizar la ruta</p>
        <p className='error-text'>{error}</p>
      </div>
    </div>
  )
}
