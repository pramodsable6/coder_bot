from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="microsoft/unixcoder-base"
)

# Load existing database
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Query semantic meaning instead of exact keywords
query = "miniconda installation on linux"
results = db.similarity_search(query, k=3)

for idx, doc in enumerate(results):
    print(f"\n--- Result {idx+1} Match from {doc.metadata['source']} ---")
    print(doc.page_content[:300] + "...")
