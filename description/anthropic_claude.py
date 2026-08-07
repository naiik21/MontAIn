import logging

import anthropic

import config

logger = logging.getLogger("montain.claude")

_client = None


def get_client():
    """
    Devuelve el cliente de Anthropic, creandolo una sola vez.

    La comprobacion anterior (`if not client`) no validaba nada: un cliente
    recien construido siempre es truthy, asi que una API key ausente pasaba
    desapercibida hasta fallar dentro de la llamada. Aqui se valida la clave.
    """
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY no esta configurado")
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def generate_description(prompt):
    # thinking desactivado a proposito: en los modelos actuales viene activado
    # por defecto y sus tokens cuentan contra max_tokens, asi que una tarea de
    # redaccion corta podria gastar el presupuesto en razonar y truncar el
    # texto. Aqui no aporta: el analisis ya viene calculado en el prompt.
    message = get_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.ANTHROPIC_MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    # Un texto cortado a media frase se ve feo y no siempre es evidente al
    # revisar la salida: conviene que quede en el log.
    if message.stop_reason == "max_tokens":
        logger.warning(
            "La descripcion se ha truncado en %s tokens. Sube ANTHROPIC_MAX_TOKENS "
            "o acorta la longitud que pide el prompt.",
            config.ANTHROPIC_MAX_TOKENS,
        )

    # Se filtra por tipo de bloque en vez de asumir que el primero es texto:
    # si algun dia se reactiva el thinking, content[0] seria el razonamiento.
    return "".join(block.text for block in message.content if block.type == "text")
