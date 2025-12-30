from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

import pickle
import os

# Path where chunks were saved
CHUNKS_PATH = "data/chunks/chunks.pkl"
VECTOR_DB_PATH = "vectorstore/chroma_db"

# Load chunks
with open(CHUNKS_PATH, "rb") as f:
    documents = pickle.load(f)

print(f"Loaded {len(documents)} document chunks")

# Load embedding model
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH
)

# Save to disk
vectorstore.persist()

print("Vector store created and saved successfully.")
