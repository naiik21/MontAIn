import { useState } from 'react'
import '../style/index.css'
import { FileUpload } from './FileUpload.jsx'
import { LoadingIndicator } from './LoadingIndicator.jsx'
import { ErrorMessage } from './ErrorMessage.jsx'
import { GPXDetails } from './GPXDetails.jsx'
import { useGPXProcessor } from '../hooks/useGPXProcessor.js'
import { MapFrame } from './MapFrame.jsx'
import { ElevationChart } from './ElevationChart.jsx'

/**
 * Componente principal Welcome que orquesta todos los subcomponentes
 */
export function Welcome() {
  const [selectedFile, setSelectedFile] = useState(null)
  const {
    isLoading,
    error,
    status,
    gpxData,
    mapHtml,
    elevationPlot,
    processFile,
    clearError
  } = useGPXProcessor()

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    clearError() // Limpiar errores anteriores al seleccionar un nuevo archivo
  }

  const handleError = (errorMessage) => {
    // Mostrar error en la interfaz
    console.error(errorMessage)
  }

  const handleProcess = async () => {
    if (selectedFile) {
      await processFile(selectedFile)
    }
  }

  return (
    <div className='container'>
      <h1>MontAIn</h1>
      <p className='subtitle'>
        Analiza rutas de montaña con inteligencia artificial
      </p>

      <FileUpload
        onFileSelect={handleFileSelect}
        onError={handleError}
        status={status}
        handleProcess={handleProcess}
        selectedFile={selectedFile}
        isLoading={isLoading}
      />

      <LoadingIndicator isLoading={isLoading} />
      <ErrorMessage error={error} />
      <div className='section data-section'>
        <GPXDetails gpxData={gpxData} />
        <div>
          <MapFrame mapHtml={mapHtml} />
          <ElevationChart elevationData={elevationPlot} />
        </div>
      </div>
    </div>
  )
}
