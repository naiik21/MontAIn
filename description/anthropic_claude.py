import os
import anthropic

def generate_description(prompt):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return message.content

