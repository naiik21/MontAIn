import { useState } from 'react'

const API_URL = 'http://localhost:8000/process-gpx'

/**
 * Hook personalizado para procesar archivos GPX
 */
export function useGPXProcessor() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState('Sube un archivo GPX y pulsa "Procesar ruta".')
  const [gpxData, setGpxData] = useState(null)
  const [mapHtml, setMapHtml] = useState(null)
  const [elevationPlot, setElevationPlot] = useState(null)
  const clearError = () => {
    setError(null)
  }

  const processFile = async (file) => {
    if (!file) {
      setError('Por favor, selecciona un archivo GPX')
      return
    }

    setIsLoading(true)
    setError(null)
    setStatus('Enviando archivo al servidor...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Error procesando el archivo')
      }

      const data = await response.json()
      setStatus('Ruta procesada correctamente en el servidor.')

      const processedData = data.data[0]
      const mapHtml = data.map_html
      const elevationPlot = data.elevation_plot
      setGpxData(processedData)
      setMapHtml(mapHtml)
      setElevationPlot(elevationPlot)
    } catch (err) {
      const errorMessage =
        err.message ||
        'Error al procesar el archivo. Asegúrate de que el servidor esté corriendo en http://localhost:8000'
      setError(errorMessage)
      setStatus('Ha ocurrido un error al procesar la ruta.')
    } finally {
      setIsLoading(false)
    }
  }

  return {
    isLoading,
    error,
    status,
    gpxData,
    mapHtml,
    elevationPlot,
    processFile,
    clearError
  }
}

