import os
import anthropic

def generate_description(prompt):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if not client:
        raise ValueError("ANTHROPIC_API_KEY no está configurado")

    return client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text
