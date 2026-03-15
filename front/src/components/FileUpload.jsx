import { useState, useRef } from 'react'
import '../style/index.css'

/**
 * Componente para subir archivos GPX con drag & drop
 */
export function FileUpload({ onFileSelect, onError, status, handleProcess, selectedFile, isLoading }) {
  const [fileName, setFileName] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileChange = (file) => {
    if (file && file.name.endsWith('.gpx')) {
      setFileName(file.name)
      onFileSelect(file)
    } else {
      onError('Por favor, selecciona un archivo GPX')
    }
  }

  const handleInputChange = (e) => {
    if (e.target.files.length > 0) {
      handleFileChange(e.target.files[0])
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0])
    }
  }

  return (
    <div
      className={`upload-section ${isDragging ? 'dragover' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        id="gpxFile"
        accept=".gpx"
        onChange={handleInputChange}
        style={{ display: 'none' }}
      />
      <label htmlFor="gpxFile" className="file-label">
        Seleccionar archivo GPX
      </label>
      {fileName && (
        <div className="file-name">Archivo seleccionado: {fileName}</div>
      )}

       <button
        id="upload"
        onClick={handleProcess}
        disabled={!selectedFile || isLoading}
      >
        Procesar ruta
      </button>

      <div style={{ fontSize: '14px', color: '#666666' }}>{status}</div>
    </div>
  )
}

