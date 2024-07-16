from groq import Groq
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone

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

# Preparing Podcast Files
mp3_file_folder = "mp3-files"
mp3_chunk_folder = "mp3-chunks"
chunk_length_ms = 1000000
overlap_ms = 10000

def split_m4a(mp3_file_folder, mp3_chunk_folder, episode_id, chunk_length_ms, overlap_ms, print_output):
    audio = AudioSegment.from_file(mp3_file_folder + "/" + episode_id + ".mp3", format="mp3")
    num_chunks = len(audio) // (chunk_length_ms - overlap_ms) + (1 if len(audio) % chunk_length_ms else 0)
    
    for i in range(num_chunks):
        start_ms = i * chunk_length_ms - (i * overlap_ms)
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]
        export_fp = mp3_chunk_folder + "/" + episode_id + f"_chunk{i+1}.mp3"
        chunk.export(export_fp, format="mp3")
        if print_output:
            print('Exporting', export_fp)
        
    return chunk

print_output = True
for fil in os.listdir(mp3_file_folder):
    episode_id = fil.split('.')[0]
    print('Splitting Episode ID:', episode_id)
    chunk = split_m4a(mp3_file_folder, mp3_chunk_folder, episode_id, chunk_length_ms, overlap_ms, print_output)
    print_output = False


# Loading Episode Metadata
episode_metadata_df = pd.read_csv('episode_metadata.csv')
chunk_fps = os.listdir(mp3_chunk_folder)
episode_chunk_df = pd.DataFrame({
    'filepath': [mp3_chunk_folder + '/' + fp for fp in chunk_fps],
    'episode_id': [fp.split('_chunk')[0] for fp in chunk_fps]
    }
)
episodes_df = episode_chunk_df.merge(episode_metadata_df, on='episode_id')
print(episodes_df.head(10))

# Transcribing
text_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)

documents = []
cnt = 0
for index, row in episodes_df.iterrows():
    cnt += 1
    audio_filepath = row['filepath']
    transcript = audio_to_text(audio_filepath)
    chunks = text_splitter.split_text(transcript)
    for chunk in chunks:
        header = f"Date: {row['published_date']}\nEpisode Title: {row['title']}\n\n"
        documents.append(Document(page_content=header + chunk, metadata={"source": "local"}))
    if np.mod(cnt ,round(len(episodes_df) / 5)) == 0:
        print(round(cnt / len(episodes_df),2) * 100, '% of transcripts processed...')
        
print('# Transcription Chunks: ', len(documents))

# Storing Transcriptions in Pinecone
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
pinecone_index_name = "semantic-search"
docsearch = PineconeVectorStore.from_documents(documents, embedding_function, index_name=pinecone_index_name)

# Querying the Vector Database
user_question = "What are state in Reactjs"
relevent_docs = docsearch.similarity_search(user_question)
relevant_transcripts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevent_docs[:3]])
transcript_chat_completion(client, relevant_transcripts, user_question)