import { useState } from 'react'
import { authClient } from './auth'

/**
 * Componente de login usando Better Auth
 * Ajusta los proveedores para que coincidan con la configuración de tu backend.
 */
export function Login() {
  const [loadingProvider, setLoadingProvider] = useState(null)
  const [error, setError] = useState(null)

  const handleSocialLogin = async (provider) => {
    try {
      setLoadingProvider(provider)
      setError(null)

      await authClient.signIn.social({
        provider,
        callbackURL: '/',
      })
    } catch (err) {
      console.error(err)
      setError('No se pudo iniciar sesión. Inténtalo de nuevo.')
    } finally {
      setLoadingProvider(null)
    }
  }

  return (
    <div className="container">
      <h1>MontAIn</h1>
      <p className="subtitle">Inicia sesión para analizar tus rutas de montaña</p>

      <div className="section">
        <button
          className="primary-button"
          onClick={() => handleSocialLogin('google')}
          disabled={loadingProvider !== null}
        >
          {loadingProvider === 'google' ? 'Conectando con Google...' : 'Iniciar sesión con Google'}
        </button>

        <button
          className="secondary-button"
          onClick={() => handleSocialLogin('github')}
          disabled={loadingProvider !== null}
          style={{ marginLeft: '1rem' }}
        >
          {loadingProvider === 'github' ? 'Conectando con GitHub...' : 'Iniciar sesión con GitHub'}
        </button>
      </div>

      {error && <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>}
    </div>
  )
}


