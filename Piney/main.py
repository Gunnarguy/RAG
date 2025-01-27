import json
from embeddings import embed_texts
from pinecone_utils import create_index, upsert_vectors, query_index

# Step 1: Load data
with open("/Users/gunnarhostetler/Documents/GitHub/RAG/Piney/data/data/sample_data.json", "r") as file:
    data = json.load(file)

# Step 2: Embed texts
texts = [item["text"] for item in data]
embeddings = embed_texts(texts)

# Step 3: Prepare vectors for Pinecone
vectors = [
    {
        "id": item["id"],
        "values": embedding,
        "metadata": {"text": item["text"]}
    }
    for item, embedding in zip(data, embeddings)
]

# Step 4: Create Pinecone index
index_name = "pineyindex"
dimension = 3072  # text-embedding-3-large dimension
index = create_index(index_name, dimension)

# Step 5: Upsert vectors
namespace = "ns1"
upsert_vectors(index, vectors, namespace)
print("Vectors upserted successfully.")

# Step 6: Query Pinecone
query_text = "Tell me about the tech company known as Apple"
query_embedding = embed_texts([query_text])[0]
results = query_index(index, query_embedding, namespace, top_k=3)

# Step 7: Print results
print("Query Results:")
for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Text: {match['metadata']['text']}")
    print()