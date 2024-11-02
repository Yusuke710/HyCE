# text_embedding.py

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss

def embed_corpus(corpus, bi_encoder, batch_size=64):
    """
    Embeds the corpus using the bi-encoder model.
    """
    corpus_chunks = [item['chunk'] for item in corpus]
    corpus_embeddings = bi_encoder.encode(
        corpus_chunks,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return corpus_embeddings

def save_embeddings(embeddings, file_path):
    """
    Saves the embeddings to a file using NumPy's save function.
    """
    embeddings_np = embeddings.cpu().numpy()
    np.save(file_path, embeddings_np)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(file_path):
    """
    Loads embeddings from a file.
    """
    if not os.path.exists(file_path):
        print(f"The embeddings file {file_path} does not exist.")
        return None
    embeddings_np = np.load(file_path)
    embeddings = torch.tensor(embeddings_np)
    print(f"Embeddings loaded from {file_path}")
    return embeddings

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from the embeddings.
    """
    embeddings = embeddings.cpu().numpy()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize embeddings
    index.add(embeddings)
    return index
