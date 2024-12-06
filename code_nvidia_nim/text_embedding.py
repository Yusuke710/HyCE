# text_embedding.py

import os
import numpy as np
import torch
import faiss
import openai

def embed_corpus(corpus, client, model, batch_size=64, input_type="passage"):
    """
    Embeds the corpus using the NVIDIA embedding model via OpenAI API.
    """
    if input_type not in ["passage", "query"]:
        raise ValueError("input_type must be either 'passage' or 'query'.")

    corpus_chunks = [item['chunk'] for item in corpus]
    print(f"Corpus chunks length: {len(corpus_chunks)}")
    embeddings_list = []

    
    # Since the API may have limitations on the batch size, we can batch the inputs
    for i in range(0, len(corpus_chunks), batch_size):
        batch_chunks = corpus_chunks[i:i+batch_size]
        response = client.embeddings.create(
            input=batch_chunks,
            model=model,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"}
        )

        # Extract embeddings from response
        for data in response.data:
            embeddings_list.append(data.embedding)
    
    # Convert to tensor
    embeddings = torch.tensor(embeddings_list)
    return embeddings


def save_embeddings(embeddings, file_path):
    """
    Saves the embeddings to a file using NumPy's save function.
    """
    embeddings_np = np.array(embeddings) if isinstance(embeddings, list) else embeddings.cpu().numpy()
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
    embeddings = np.array(embeddings) if isinstance(embeddings, list) else embeddings.cpu().numpy()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize embeddings
    index.add(embeddings)
    return index
