
export function MapFrame({ mapHtml }) {
  if (!mapHtml) return null

  return (
    <div
    className="section"
      style={{ marginTop: '24px' }}
      dangerouslySetInnerHTML={{ __html: mapHtml }}
    />
  )
}