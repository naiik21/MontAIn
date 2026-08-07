import { useRef, useState } from 'react'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * Zona de subida con arrastrar-y-soltar. Dos pasos, como el backend:
 * elegir el archivo y despues analizarlo.
 */
export function FileUpload({ selectedFile, onFileSelect, onError, onProcess }) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

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
            <button type='button' className='btn btn--primary' onClick={onProcess}>
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
        </>
      )}

      <span className='dropzone-hint'>máx. 5 MB · solo .gpx</span>
    </div>
  )
}
