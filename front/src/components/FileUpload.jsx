import { useRef, useState } from 'react'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/*
 * Trazas de muestra para quien llega sin un GPX a mano. Se sirven como
 * ficheros estaticos y se envian por el mismo camino que una subida normal,
 * asi que aprovechan la cache del backend: solo la primera visita paga el
 * analisis y la llamada a Claude.
 */
const EXAMPLES = [
  { file: 'almanzor.gpx', name: 'Almanzor', detail: '8,2 km · +1789 m' },
  { file: 'olivos-centenarios.gpx', name: 'Olivos Centenarios', detail: '6,7 km · +240 m' },
]

/**
 * Zona de subida con arrastrar-y-soltar. Dos pasos, como el backend:
 * elegir el archivo y despues analizarlo.
 */
export function FileUpload({ selectedFile, onFileSelect, onError, onProcess }) {
  const [isDragging, setIsDragging] = useState(false)
  const [loadingExample, setLoadingExample] = useState(null)
  const inputRef = useRef(null)

  const loadExample = async (example) => {
    setLoadingExample(example.file)
    try {
      const res = await fetch(`/ejemplos/${example.file}`)
      if (!res.ok) throw new Error()
      const blob = await res.blob()
      // Se envuelve en un File para que siga exactamente el mismo camino
      // que un archivo elegido por el usuario.
      onProcess(new File([blob], example.file, { type: 'application/gpx+xml' }))
    } catch {
      onError('No se ha podido cargar la ruta de ejemplo.')
    } finally {
      setLoadingExample(null)
    }
  }

  const handleFile = (file) => {
    if (file && file.name.toLowerCase().endsWith('.gpx')) {
      onFileSelect(file)
    } else {
      onError('El archivo debe tener extensión .gpx')
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0])
  }

  return (
    <div
      className={`dropzone ${isDragging ? 'dropzone--active' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type='file'
        accept='.gpx'
        hidden
        onChange={(e) => {
          if (e.target.files.length > 0) handleFile(e.target.files[0])
          e.target.value = '' // permite volver a elegir el mismo archivo
        }}
      />

      <span className='dropzone-ext'>.gpx</span>

      {selectedFile ? (
        <>
          <div className='filechip'>
            <span className='filechip-name'>{selectedFile.name}</span>
            <span className='filechip-size'>{formatSize(selectedFile.size)}</span>
          </div>
          <div className='dropzone-actions'>
            <button
              type='button'
              className='btn btn--primary'
              onClick={() => onProcess(selectedFile)}
            >
              Analizar ruta
            </button>
            <button
              type='button'
              className='link-quiet'
              onClick={() => inputRef.current?.click()}
            >
              cambiar archivo
            </button>
          </div>
        </>
      ) : (
        <>
          <p className='dropzone-text'>Arrastra aquí la traza de tu ruta</p>
          <div className='dropzone-actions'>
            <button
              type='button'
              className='btn btn--primary'
              onClick={() => inputRef.current?.click()}
            >
              Elegir archivo
            </button>
          </div>

          <div className='examples'>
            <span className='examples-label'>o prueba con una ruta de ejemplo</span>
            <div className='examples-list'>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.file}
                  type='button'
                  className='example'
                  onClick={() => loadExample(ex)}
                  disabled={loadingExample !== null}
                >
                  <span className='example-name'>{ex.name}</span>
                  <span className='example-detail'>{ex.detail}</span>
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      <span className='dropzone-hint'>máx. 5 MB · solo .gpx</span>
    </div>
  )
}
