import { useState } from 'react'
import '../style/index.css'
import { DifficultyScale } from './DifficultyScale.jsx'
import { ElevationChart } from './ElevationChart.jsx'
import { ErrorMessage } from './ErrorMessage.jsx'
import { FileUpload } from './FileUpload.jsx'
import { LoadingIndicator } from './LoadingIndicator.jsx'
import { MapFrame } from './MapFrame.jsx'
import { MetricsGrid } from './MetricsGrid.jsx'
import { RouteDescription } from './RouteDescription.jsx'
import { ThemeToggle } from './ThemeToggle.jsx'
import { Waymark } from './Waymark.jsx'
import { useGPXProcessor } from '../hooks/useGPXProcessor.js'

const fmt = (n, digits = 0) =>
  new Intl.NumberFormat('es-ES', { maximumFractionDigits: digits }).format(n)

function SectionTitle({ children }) {
  return (
    <h2 className='section-title'>
      <Waymark variant='section' />
      {children}
    </h2>
  )
}

export function Welcome() {
  const [selectedFile, setSelectedFile] = useState(null)
  // Errores de validacion en el cliente (extension incorrecta), separados de
  // los errores que devuelve la API.
  const [localError, setLocalError] = useState(null)
  const {
    isLoading,
    error,
    gpxData,
    mapHtml,
    elevationPlot,
    description,
    processFile,
    clearError,
    reset
  } = useGPXProcessor()

  const handleReset = () => {
    reset()
    setSelectedFile(null)
    setLocalError(null)
  }

  const shownError = error || localError

  return (
    <div className='app'>
      <header className='site-header'>
        <a className='brand' href='/'>
          <Waymark variant='brand' />
          <span className='brand-name'>MontAIn</span>
        </a>
        <div className='header-actions'>
          {gpxData && (
            <button
              type='button'
              className='btn btn--ghost'
              onClick={handleReset}>
              Nueva ruta
            </button>
          )}
          <ThemeToggle />
        </div>
      </header>

      <main>
        {!gpxData ? (
          /* ---- estado inicial: el uploader es el protagonista ---- */
          <section className='hero'>
            <span className='eyebrow'>Análisis de rutas de montaña</span>
            <h1 className='hero-title'>Lee el terreno antes de pisarlo.</h1>
            <p className='hero-lede'>
              Sube una traza GPX: MontAIn calcula pendientes, desnivel y
              exposición, estima la dificultad con un modelo entrenado sobre
              miles de rutas y redacta una descripción de guía con IA.
            </p>

            {isLoading ? (
              <LoadingIndicator />
            ) : (
              <FileUpload
                selectedFile={selectedFile}
                onFileSelect={(file) => {
                  setSelectedFile(file)
                  setLocalError(null)
                  clearError()
                }}
                onError={setLocalError}
                onProcess={() => processFile(selectedFile)}
              />
            )}

            <ErrorMessage error={shownError} />
          </section>
        ) : (
          /* ---- ficha de ruta ---- */
          <article className='dossier'>
            <header className='dossier-header'>
              <span className='eyebrow'>Ficha de ruta</span>
              <h1 className='route-name'>
                {gpxData.filename || 'Ruta sin nombre'}
              </h1>
              <div className='route-vitals'>
                <DifficultyScale difficulty={gpxData.difficulty} />
                <div className='route-stats'>
                  <span>
                    <strong>{fmt(gpxData.distance_km, 1)}</strong> km
                  </span>
                  <span>
                    <strong>+{fmt(gpxData.elevation_gain)}</strong> m
                  </span>
                  <span>
                    máx. <strong>{fmt(gpxData.max_elevation)}</strong> m
                  </span>
                </div>
              </div>
            </header>

            <div className='dossier-stack'>
              <section className='panel'>
                <SectionTitle>Recorrido</SectionTitle>
                <MapFrame mapHtml={mapHtml} />
              </section>
              <section className='panel'>
                <SectionTitle>Perfil de elevación</SectionTitle>
                <ElevationChart elevationData={elevationPlot} />
              </section>
            </div>

            <section className='metrics-section'>
              <SectionTitle>Métricas</SectionTitle>
              <MetricsGrid record={gpxData} />
            </section>

            {description && (
              <section className='description-section'>
                <SectionTitle>Descripción de guía</SectionTitle>
                <RouteDescription text={description} />
                <p className='description-note'>
                  Escrita con IA a partir de los datos de la traza. Contrasta
                  siempre con fuentes locales antes de salir.
                </p>
              </section>
            )}
          </article>
        )}
      </main>

      <footer className='site-footer'>
        <span>MontAIn — análisis de rutas GPX</span>
        <span>
          FastAPI · XGBoost · Astro ·{' '}
          <a
            href='https://github.com/naiik21/MontAIn'
            target='_blank'
            rel='noopener noreferrer'>
            GitHub
          </a>
        </span>
      </footer>
    </div>
  )
}
