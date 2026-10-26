import chromadb
import pandas as pd

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("langchain")

data = collection.get(include=["documents", "metadatas", "embeddings"])

df = pd.DataFrame({
    "id": data["ids"],
    "text": data["documents"],
    "metadata": data["metadatas"],
})

df["source"] = df["metadata"].apply(lambda item: item.get("source") if item else None)
df["filename"] = df["metadata"].apply(lambda item: item.get("filename") if item else None)
df = df.drop(columns="metadata")

print(df.head())
print(df.shape)
import pdb; pdb.set_trace()