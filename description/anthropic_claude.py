import anthropic

import config

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
    return get_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.ANTHROPIC_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text
