import { useState } from 'react'

// Configurable en despliegue con PUBLIC_API_URL (Astro expone al cliente las
// variables con prefijo PUBLIC_). En local basta con no definirla.
const API_BASE = import.meta.env.PUBLIC_API_URL ?? 'http://localhost:8000'
const API_URL = `${API_BASE.replace(/\/$/, '')}/process-gpx`

/**
 * Estado y llamada al analisis de una traza GPX.
 */
export function useGPXProcessor() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [gpxData, setGpxData] = useState(null)
  const [mapHtml, setMapHtml] = useState(null)
  const [elevationPlot, setElevationPlot] = useState(null)
  const [description, setDescription] = useState(null)

  const clearError = () => setError(null)

  /** Vuelve al estado inicial ("Nueva ruta"). */
  const reset = () => {
    setGpxData(null)
    setMapHtml(null)
    setElevationPlot(null)
    setDescription(null)
    setError(null)
  }

  const processFile = async (file) => {
    if (!file) {
      setError('Selecciona primero un archivo GPX.')
      return
    }

    setIsLoading(true)
    setError(null)

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
      setGpxData(data.data[0])
      setMapHtml(data.map_html)
      setElevationPlot(data.elevation_plot)
      setDescription(data.description ?? null)
    } catch (err) {
      // fetch solo lanza cuando la peticion no llega a salir (servidor caido,
      // CORS, sin red); su mensaje generico no le dice nada al usuario.
      const isNetworkError = err instanceof TypeError
      const errorMessage = isNetworkError
        ? `No se ha podido contactar con el servidor (${API_BASE}). Comprueba que esté en marcha.`
        : err.message || 'Error al procesar el archivo.'
      setError(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  return {
    isLoading,
    error,
    gpxData,
    mapHtml,
    elevationPlot,
    description,
    processFile,
    clearError,
    reset
  }
}
