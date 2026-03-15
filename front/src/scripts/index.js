/**
 * Archivo principal que inicializa la aplicación
 */

import { getDOMElements, showError } from './utils.js'
import { setupFileInput, setupDragAndDrop } from './fileHandler.js'
import { processGPXFile } from './gpxProcessor.js'

// Obtener referencias a elementos DOM
const elements = getDOMElements()

// Función wrapper para showError con los elementos correctos
const handleError = (message) => {
  showError(message, elements.error, elements.statusMessage)
}

// Configurar manejo de archivos
setupFileInput(
  elements.fileInput,
  elements.fileName,
  elements.uploadBtn,
  handleError
)

// Configurar drag & drop
setupDragAndDrop(
  elements.uploadSection,
  elements.fileInput,
  elements.fileName,
  elements.uploadBtn,
  handleError
)

// Configurar botón de procesamiento
elements.uploadBtn.addEventListener('click', async () => {
  await processGPXFile(elements.fileInput.files[0], elements, handleError)
})

