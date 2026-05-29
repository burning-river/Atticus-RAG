from huggingface_hub import InferenceClient
import numpy as np
# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")

embedding_model_name = 'BAAI/bge-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

os.environ["HF_TOKEN"] = HF_TOKEN
client = InferenceClient()

def embed_documents(documents: list[str]) -> np.ndarray:
    # return embedding_model.encode(documents, convert_to_numpy=True)
    embeddings = client.feature_extraction(
    text=documents,
    model=embedding_model_name,
)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    # return embedding_model.encode([query], convert_to_numpy=True)[0]
    embedding = client.feature_extraction(
    text=query,
    model=embedding_model_name,
)
    return embedding
