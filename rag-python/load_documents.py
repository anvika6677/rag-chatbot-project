from langchain_community.document_loaders import PyPDFLoader
import os

DATA_DIR = "data/docs"

def load_pdfs(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {file} with {len(docs)} pages")
    return documents

if __name__ == "__main__":
    docs = load_pdfs(DATA_DIR)
    print(f"\nTotal pages loaded: {len(docs)}")
