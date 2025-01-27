import pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('/Users/gunnarhostetler/Documents/GitHub/RAG/Piney/.env')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Initialize the Pinecone client
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key=os.environ.get('PINECONE_API_KEY')
)

def create_index(name, dimension):
    if name not in pinecone.list_indexes():
        pinecone.create_index(name=name, dimension=dimension)

def cleanup_index(name):
    if name in pinecone.list_indexes():
        pinecone.delete_index(name)

def upsert_vectors(index_name, vectors):
    index = pinecone.Index(index_name)
    index.upsert(vectors)

def query_index(index_name, vector, top_k=10):
    index = pinecone.Index(index_name)
    query_result = index.query(vector=vector, top_k=top_k)
    return query_result
