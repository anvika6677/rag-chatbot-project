from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

DOCS_PATH = "data/docs"

def load_and_split_documents():
    documents = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            file_path = os.path.join(DOCS_PATH, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    chunks = load_and_split_documents()
    print(f"Total chunks created: {len(chunks)}")
    print("Sample chunk:\n")
    print(chunks[0].page_content[:500])
import pickle
import os

os.makedirs("data/chunks", exist_ok=True)

with open("data/chunks/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"Saved {len(chunks)} chunks to data/chunks/chunks.pkl")
