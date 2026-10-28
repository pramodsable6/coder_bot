import chromadb
import pandas as pd

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("langchain")

# Get total number of items to handle batching
total_count = collection.count()
print(f"Total records in collection: {total_count}")

# Fetch data in batches to avoid SQLite variable limits
batch_size = 500  
all_dfs = []

for i in range(0, total_count, batch_size):
    # Fetch a slice of IDs first
    all_ids = collection.get(limit=batch_size, offset=i)["ids"]
    if not all_ids:
        break
        
    # Get complete documents for this specific batch
    data = collection.get(
        ids=all_ids, 
        include=["documents", "metadatas", "embeddings"]
    )
    
    batch_df = pd.DataFrame({
        "id": data["ids"],
        "text": data["documents"],
        "metadata": data["metadatas"],
    })
    all_dfs.append(batch_df)

# Combine batches into a final DataFrame
if all_dfs:
    df = pd.concat(all_dfs, ignore_index=True)
    df["source"] = df["metadata"].apply(lambda item: item.get("source") if item else None)
    df["filename"] = df["metadata"].apply(lambda item: item.get("filename") if item else None)
    df = df.drop(columns="metadata")
    
    print("\n--- DataFrame Head ---")
    print(df.head())
    print(f"\nFinal Shape: {df.shape}")
else:
    print("No data found in the collection.")

import pdb; pdb.set_trace()
