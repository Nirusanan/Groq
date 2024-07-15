from groq import Groq
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

client = Groq(api_key = os.getenv('GROQ_API_KEY'))
model = 'whisper-large-v3'


# Audio to Text
def audio_to_text(filepath):
    with open(filepath, "rb") as file:
        translation = client.audio.translations.create(
            file=(filepath, file.read()),
            model="whisper-large-v3",
        )
    return translation.text

filepath = "RouteLLM.mp3"
translation_text = audio_to_text(filepath)
print(translation_text[:2000])

# Ask Questions from Audio Transcripts
def transcript_chat_completion(client, transcript, user_question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": '''Use this transcript or transcripts to answer any user questions, citing specific quotes:

                {transcript}
                '''.format(transcript=transcript)
            },
            {
                "role": "user",
                "content": user_question,
            }
        ],
        model="llama3-70b-8192",
    )
    print(chat_completion.choices[0].message.content)

user_question = "What is the meaning of Route LLM"
transcript_chat_completion(client, translation_text, user_question)  