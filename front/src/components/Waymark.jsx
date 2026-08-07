/**
 * Marca de sendero GR: barra blanca sobre barra roja.
 * Es la marca de la casa; la variante `section` señaliza cada
 * sección de la ficha igual que las marcas guían en el monte.
 */
export function Waymark({ variant }) {
  const cls = variant ? `waymark waymark--${variant}` : 'waymark'
  return (
    <span className={cls} aria-hidden='true'>
      <span className='waymark-top' />
      <span className='waymark-bottom' />
    </span>
  )
}
