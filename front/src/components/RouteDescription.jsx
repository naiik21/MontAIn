/**
 * Renderiza la descripcion que escribe Claude sin depender de una libreria
 * de markdown: el texto usa un subconjunto pequeno (titulos #, **negritas**
 * y listas con -), y React escapa el contenido por defecto, asi que
 * construir nodos es seguro y suficiente.
 */

function inline(text, keyBase) {
  // **negrita** y *cursiva*. Se capturan los delimitadores para poder
  // distinguirlos al reconstruir; sin esto los asteriscos de la nota final
  // que escribe el modelo se verian literales.
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g)
  return parts.map((part, i) => {
    const key = `${keyBase}-${i}`
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return <em key={key}>{part.slice(1, -1)}</em>
    }
    return part
  })
}

function parse(text) {
  const blocks = []
  let list = null

  const lines = text.split('\n')
  for (const raw of lines) {
    const line = raw.trim()

    if (/^[-•]\s+/.test(line)) {
      if (!list) {
        list = []
        blocks.push({ type: 'ul', items: list })
      }
      list.push(line.replace(/^[-•]\s+/, ''))
      continue
    }
    list = null

    if (line === '') continue

    const heading = line.match(/^#{1,4}\s+(.*)/)
    if (heading) {
      blocks.push({ type: 'h', text: heading[1] })
    } else {
      blocks.push({ type: 'p', text: line })
    }
  }

  // El primer titulo suele ser "Descripción de la ruta", que duplicaria
  // el titulo de la seccion: se omite.
  if (blocks[0]?.type === 'h') blocks.shift()

  return blocks
}

export function RouteDescription({ text }) {
  if (!text) return null

  return (
    <div className='description-body'>
      {parse(text).map((block, i) => {
        if (block.type === 'h') return <h4 key={i}>{inline(block.text, i)}</h4>
        if (block.type === 'ul')
          return (
            <ul key={i}>
              {block.items.map((item, j) => (
                <li key={j}>{inline(item, `${i}-${j}`)}</li>
              ))}
            </ul>
          )
        return <p key={i}>{inline(block.text, i)}</p>
      })}
    </div>
  )
}
