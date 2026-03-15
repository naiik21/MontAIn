/**
 * Procesamiento de archivos GPX y comunicación con la API
 */

const API_URL = 'http://localhost:8000/process-gpx'

/**
 * Genera el HTML con los detalles del GPX procesado
 * @param {Object} gpxData - Datos del GPX procesado
 * @returns {string} HTML formateado
 */
function formatGPXDetails(gpxData) {
  return `
    <p><strong>Nombre GPX interno:</strong> ${gpxData.filename || 'N/A'}</p>
    <p><strong>Distance:</strong> ${gpxData.distance_km ?? 'N/A'} km</p>
    <p><strong>Elevation Gain:</strong> ${gpxData.elevation_gain ?? 'N/A'} m</p>
    <p><strong>Elevation Loss:</strong> ${gpxData.elevation_loss ?? 'N/A'} m</p>
    <p><strong>Max Elevation:</strong> ${gpxData.max_elevation ?? 'N/A'} m</p>
    <p><strong>Min Elevation:</strong> ${gpxData.min_elevation ?? 'N/A'} m</p>
    <p><strong>Max Slope:</strong> ${gpxData.max_slope ?? 'N/A'}°</p>
    <p><strong>Mean Slope:</strong> ${gpxData.mean_slope ?? 'N/A'}°</p>
    <p><strong>% Over 30°:</strong> ${gpxData.pct_over_30 != null ? (gpxData.pct_over_30 * 100).toFixed(1) + '%' : 'N/A'}</p>
    <p><strong>% Over 40°:</strong> ${gpxData.pct_over_40 != null ? (gpxData.pct_over_40 * 100).toFixed(1) + '%' : 'N/A'}</p>
    <p><strong>% Over 45°:</strong> ${gpxData.pct_over_45 != null ? (gpxData.pct_over_45 * 100).toFixed(1) + '%' : 'N/A'}</p>
    <p><strong>Mean Aspect:</strong> ${gpxData.mean_aspect ?? 'N/A'}°</p>
    <p><strong>Rugosity Mean:</strong> ${gpxData.rugosity_mean ?? 'N/A'}</p>
    <p><strong>Exposed %:</strong> ${gpxData.exposed_pct != null ? (gpxData.exposed_pct * 100).toFixed(1) + '%' : 'N/A'}</p>
    <p><strong>Difficulty:</strong> ${gpxData.difficulty || 'N/A'}</p>
  `
}

/**
 * Procesa el archivo GPX enviándolo al servidor
 * @param {File} file - Archivo GPX a procesar
 * @param {Object} elements - Objeto con referencias a elementos DOM
 * @param {Function} onError - Función para manejar errores
 */
export async function processGPXFile(file, elements, onError) {
  const { loading, error, uploadBtn, statusMessage, gpxDetails } = elements

  if (!file) {
    onError('Por favor, selecciona un archivo GPX')
    return
  }

  const formData = new FormData()
  formData.append('file', file)

  // Mostrar loading
  loading.classList.add('active')
  error.classList.remove('active')
  uploadBtn.disabled = true
  statusMessage.textContent = 'Enviando archivo al servidor...'

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
    statusMessage.textContent = 'Ruta procesada correctamente en el servidor.'

    console.log(data)

    const gpxData = data[1]

    console.log(gpxData)

    // Mostrar detalles del GPX
    gpxDetails.innerHTML = formatGPXDetails(gpxData)
  } catch (err) {
    onError(
      err.message ||
        'Error al procesar el archivo. Asegúrate de que el servidor esté corriendo en http://localhost:8000'
    )
  } finally {
    loading.classList.remove('active')
    uploadBtn.disabled = false
  }
}

