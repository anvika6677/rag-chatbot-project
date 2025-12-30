from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

VECTOR_DB_PATH = "vectorstore/chroma_db"

# Load embedding model (same one used during indexing)
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load vector store
vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_model
)

# Sample student query
query = "What subjects are taught in 4th year AI department?"

# Retrieve top 3 relevant chunks
docs = vectorstore.similarity_search(query, k=3)

print("\nRetrieved Documents:\n")
for i, doc in enumerate(docs, 1):
    print(f"--- Document {i} ---")
    print(doc.page_content[:500])
    print()
