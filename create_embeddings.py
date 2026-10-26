import os
from pathlib import Path
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 1. Configuration
CODEBASE_PATH = "../Repos/Python"       # Path to your source code repository
DB_PERSIST_PATH = "./chroma_db"      # Where the vector store will be saved on disk
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure your API key is set

# Mapping of file extensions to their corresponding LangChain parser language
SUPPORTED_LANGUAGES = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.ts': Language.TS,
    '.go': Language.GO,
    '.cpp': Language.CPP,
    '.java': Language.JAVA,
    '.md': Language.MARKDOWN,
}

def load_and_chunk_codebase(base_path):
    """Walks through the codebase, extracts code, and chunks it intelligently."""
    documents = []
    metadatas = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in SUPPORTED_LANGUAGES:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    # Split code based on the specific language grammar structure
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=SUPPORTED_LANGUAGES[ext],
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    
                    chunks = splitter.split_text(code_content)
                    
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "source": file_path,
                            "filename": file,
                            "chunk_index": i
                        })
                except Exception as e:
                    print(f"Skipping {file_path} due to error: {e}")
                    
    return documents, metadatas

def main():
    print(f"Scanning and processing codebase at: {CODEBASE_PATH}...")
    docs, meta = load_and_chunk_codebase(CODEBASE_PATH)
    print(f"Generated {len(docs)} chunks from the codebase.")
    
    # 2. Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="microsoft/unixcoder-base"
    )
    
    # 3. Build and persist the Vector Database
    print("Generating embeddings and building vector index...")
    vector_store = Chroma.from_texts(
        texts=docs,
        metadatas=meta,
        embedding=embedding_model,
        persist_directory=DB_PERSIST_PATH
    )
    print(f"Vector database successfully created and saved to '{DB_PERSIST_PATH}'")

if __name__ == "__main__":
    main()
