/**
 * Utilidades y referencias a elementos DOM
 */

export function getDOMElements() {
  return {
    fileInput: document.getElementById('gpxFile'),
    uploadBtn: document.getElementById('upload'),
    fileName: document.getElementById('fileName'),
    loading: document.getElementById('loading'),
    error: document.getElementById('error'),
    uploadSection: document.getElementById('uploadSection'),
    statusMessage: document.getElementById('statusMessage'),
    gpxDetails: document.getElementById('gpxDetails')
  }
}

/**
 * Muestra un mensaje de error en la interfaz
 * @param {string} message - Mensaje de error a mostrar
 * @param {HTMLElement} errorElement - Elemento donde mostrar el error
 * @param {HTMLElement} statusElement - Elemento donde mostrar el estado
 */
export function showError(message, errorElement, statusElement) {
  errorElement.textContent = message
  errorElement.classList.add('active')
  statusElement.textContent = 'Ha ocurrido un error al procesar la ruta.'
}

