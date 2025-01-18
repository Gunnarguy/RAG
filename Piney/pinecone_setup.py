import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# API key
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the PINECONE_API_KEY environment variable.")

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Define the serverless specification
spec = ServerlessSpec(
    cloud="aws",        # Cloud provider
    region="us-east-1"  # AWS region
)

# Create the index with specified specs
try:
    pc.create_index(
        name="pineyindex",  # Replace with your desired index name
        dimension=3072,        # Embedding dimensions
        metric="cosine",       # Distance metric
        spec=spec              # Capacity mode: Serverless
    )
    print("Index created successfully.")
except Exception as e:
    print(f"Error creating index: {e}")
