/**
 * Manejo de selección de archivos y drag & drop
 */

/**
 * Configura los event listeners para la selección de archivos
 * @param {HTMLElement} fileInput - Input de archivo
 * @param {HTMLElement} fileName - Elemento para mostrar el nombre del archivo
 * @param {HTMLElement} uploadBtn - Botón de subida
 * @param {Function} onError - Función para mostrar errores
 */
export function setupFileInput(fileInput, fileName, uploadBtn, onError) {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      fileName.textContent = `Archivo seleccionado: ${e.target.files[0].name}`
      uploadBtn.disabled = false
    }
  })
}

/**
 * Configura los event listeners para drag & drop
 * @param {HTMLElement} uploadSection - Sección de subida
 * @param {HTMLElement} fileInput - Input de archivo
 * @param {HTMLElement} fileName - Elemento para mostrar el nombre del archivo
 * @param {HTMLElement} uploadBtn - Botón de subida
 * @param {Function} onError - Función para mostrar errores
 */
export function setupDragAndDrop(uploadSection, fileInput, fileName, uploadBtn, onError) {
  uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault()
    uploadSection.classList.add('dragover')
  })

  uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover')
  })

  uploadSection.addEventListener('drop', (e) => {
    e.preventDefault()
    uploadSection.classList.remove('dragover')

    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0]
      if (file.name.endsWith('.gpx')) {
        fileInput.files = e.dataTransfer.files
        fileName.textContent = `Archivo seleccionado: ${file.name}`
        uploadBtn.disabled = false
      } else {
        onError('Por favor, selecciona un archivo GPX')
      }
    }
  })
}

