import os
import requests

def embed_texts(texts):
    headers = {
        'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY")}'
    }
    json_data = {
        'input': texts,
        'model': 'text-embedding-3-large'
    }
    response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=json_data)
    data = response.json()
    return [datum['embedding'] for datum in data['data']]
