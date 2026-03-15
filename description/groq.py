import os
from groq import Groq

def generate_description(prompt):
    
    client = Groq(
        api_key=os.environ["GROQ_API_KEY"],
    )
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content