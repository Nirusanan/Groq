import os

from groq import Groq

api_key = os.getenv('GROQ_API_KEY')

if api_key is None:
    raise ValueError("No API key found. Please set the GROQ_API_KEY environment variable.")


client = Groq(
    api_key=api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of large language models",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)