import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import json
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss
import numpy as np
import openai  # Importing openai for the LLM client
import backoff  # For handling rate limits and timeouts
import re  # For extracting JSON if needed

from llm import get_response_from_llm, extract_json_between_markers

# Existing functions for data processing and retrieval
def normalize_url(url):
    """
    Normalizes the URL by removing the fragment and query parts,
    and ensuring there is no trailing slash.
    """
    parsed = urlparse(url)
    # Remove fragment and query, and standardize the path
    path = parsed.path.rstrip('/')
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))
    return normalized

def save_web_data(url, output_dir='web_data'):
    """
    Fetches the web page from the given URL, extracts the content from linked pages,
    and saves the data to a JSON file.
    Each chunk (paragraph) is treated individually.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch the main URL {url}: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    anchors = soup.find_all('a')

    data = []
    os.makedirs(output_dir, exist_ok=True)
    visited_urls = set()

    for anchor in anchors:
        href = anchor.get('href')
        if href and not href.startswith('#'):
            full_url = urljoin(url, href)
            normalized_url = normalize_url(full_url)
            if normalized_url not in visited_urls:
                visited_urls.add(normalized_url)
                try:
                    link_response = requests.get(full_url)
                    link_response.raise_for_status()
                    link_soup = BeautifulSoup(link_response.content, 'html.parser')

                    sections = [section.get_text(strip=True) for section in link_soup.find_all('p')]
                    if sections:
                        for idx, chunk in enumerate(sections):
                            data.append({
                                'url': normalized_url,
                                'chunk_idx': idx,
                                'chunk': chunk
                            })
                        print(f'Saved content for {normalized_url}')
                    else:
                        print(f'No content found at {normalized_url}')
                except requests.RequestException as e:
                    print(f'Failed to fetch {normalized_url}: {e}')

    json_file_path = os.path.join(output_dir, 'web_data.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f'All data saved to {json_file_path}')

def load_web_data(json_file_path):
    """
    Loads the web data from a JSON file.
    """
    if not os.path.exists(json_file_path):
        print(f"The file {json_file_path} does not exist.")
        return []
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

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

def search_with_faiss(query_embedding, index, top_k):
    """
    Searches the FAISS index for the top_k most similar embeddings.
    """
    query_embedding = query_embedding.cpu().numpy()
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

def search(query, corpus, corpus_embeddings, bi_encoder, cross_encoder, index, top_k=5):
    """
    Searches for the most relevant chunks to the query.
    """
    # Encode the query using the bi-encoder
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.unsqueeze(0)  # Add batch dimension

    # Compute cosine similarities using FAISS
    distances, indices = search_with_faiss(query_embedding, index, 100)

    # Prepare cross-encoder input
    cross_inp = []
    cross_chunks = []
    for idx in indices:
        idx = int(idx)
        chunk = corpus[idx]['chunk']
        cross_inp.append((query, chunk))
        chunk_info = corpus[idx].copy()
        cross_chunks.append(chunk_info)

    # Re-rank using cross-encoder
    cross_scores = cross_encoder.predict(cross_inp)

    # Combine chunks with their cross-encoder scores
    for i, score in enumerate(cross_scores):
        cross_chunks[i]['score'] = score

    # Sort the chunks based on cross-encoder scores
    cross_chunks = sorted(cross_chunks, key=lambda x: x['score'], reverse=True)

    # Remove duplicates based on URL and chunk content
    seen = set()
    unique_chunks = []
    for chunk in cross_chunks:
        identifier = (chunk['url'], chunk['chunk'])
        if identifier not in seen:
            seen.add(identifier)
            unique_chunks.append(chunk)
        if len(unique_chunks) >= top_k:
            break

    # Return the top_k unique chunks
    return unique_chunks

# Example usage
if __name__ == "__main__":
    # URL of the document
    url = 'https://docs.restech.unsw.edu.au/'

    # Save the data
    #save_web_data(url)

    # Load the data
    corpus = load_web_data('web_data/web_data.json')

    if corpus:
        # Initialize models
        bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

        # Define the embeddings file path
        embeddings_file = 'web_data/corpus_embeddings.npy'

        # Check if embeddings file exists
        corpus_embeddings = load_embeddings(embeddings_file)

        if corpus_embeddings is None:
            # Embed the corpus
            corpus_embeddings = embed_corpus(corpus, bi_encoder)
            # Save embeddings to file
            save_embeddings(corpus_embeddings, embeddings_file)

        # Build FAISS index
        index = build_faiss_index(corpus_embeddings)

        # Define the query
        query = 'How long is my queue?' #'What GPUs are available for me?'

        # Search for relevant chunks
        top_k = 5
        top_chunks = search(
            query,
            corpus,
            corpus_embeddings,
            bi_encoder,
            cross_encoder,
            index,
            top_k=top_k
        )

        # Display the retrieved chunks
        print("\nRetrieved Chunks:")
        for i, chunk_info in enumerate(top_chunks):
            print(f"Rank {i+1}:")
            print(f"URL: {chunk_info['url']}")
            print(f"Chunk Index: {chunk_info['chunk_idx']}")
            print(f"Score: {chunk_info['score']:.4f}")
            print(f"Content: {chunk_info['chunk']}\n")

        # Prepare the context from top_chunks
        context = ""
        for i, chunk_info in enumerate(top_chunks):
            context += f"Document {i+1}:\n{chunk_info['chunk']}\n\n"

        # Prepare the LLM client and model
        # Set your OpenAI API key
        openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual API key

        client = openai
        model = 'gpt-4o-2024-08-06'  # Use an available OpenAI model

        # Note: Since the model 'gpt-3.5-turbo' is not in the provided function's supported models,
        # and you asked not to change the function, we need to select a supported model.
        # For demonstration purposes, let's assume 'gpt-4o-2024-08-06' maps to 'gpt-3.5-turbo'.

        # Prepare the system message and user message
        system_message = "You are an assistant that provides accurate and concise answers based on the provided documents."
        user_message = f"Answer the following question based on the provided documents.\n\nQuestion: {query}\n\nDocuments:\n{context}"

        # Call the LLM using the provided function
        content, new_msg_history = get_response_from_llm(
            msg=user_message,
            client=client,
            model='gpt-4o-2024-08-06',  # Using a supported model from the function
            system_message=system_message,
            print_debug=True
        )

        # Print the final answer
        print("Final Answer from LLM:")
        print(content)
    else:
        print("No data to process.")
