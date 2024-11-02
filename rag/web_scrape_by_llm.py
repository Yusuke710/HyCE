# web_scrape.py

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import json
import re
import openai
import backoff

# Import the functions provided
from llm import get_response_from_llm, extract_json_between_markers

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

def estimate_token_count(text):
    """
    Estimates the number of tokens in the text.
    """
    return len(text) / 4  # Approximate estimation

def chunk_text_semantically(text):
    """
    Uses an LLM to split the text into semantically coherent chunks.
    """
    # Prepare the prompt
    prompt = (
        "Please split the following text into semantically coherent chunks. "
        "Each chunk should represent a coherent idea or topic. "
        "Return the chunks as a JSON array of strings.\n\nText:\n" + text
    )
    # Set up client and model
    client = openai  # Replace with the appropriate client if needed
    model = 'gpt-4o-2024-08-06'  # Use a model supported by get_response_from_llm
    system_message = "You are a helpful assistant that splits text into semantically coherent chunks."

    # Call the LLM
    try:
        content, _ = get_response_from_llm(
            msg=prompt,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=False
        )
        # Extract the JSON
        chunks = extract_json_between_markers(content)
        if chunks is None:
            print("Failed to extract JSON from LLM output.")
            return []
        return chunks
    except Exception as e:
        print(f"Error during semantic chunking: {e}")
        return []

def save_web_data(url, output_dir='web_data'):
    """
    Fetches the web page from the given URL, extracts the content from linked pages,
    and saves the data to a JSON file.
    Each page's content is chunked semantically using an LLM.
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

                    full_text = link_soup.get_text(separator='\n', strip=True)
                    if estimate_token_count(full_text) > 6000:
                        print(f"Content at {normalized_url} is too long, truncating.")
                        full_text = full_text[:24000]  # Approximately 6000 tokens

                    chunks = chunk_text_semantically(full_text)
                    if chunks:
                        for idx, chunk in enumerate(chunks):
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

        # Optional: Limit the number of URLs processed for testing purposes
        # if len(visited_urls) >= 5:
        #     break

    json_file_path = os.path.join(output_dir, 'web_data.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f'All data saved to {json_file_path}')

def load_data_chunks(json_file_path):
    """
    Loads the web data from a JSON file.
    """
    if not os.path.exists(json_file_path):
        print(f"The file {json_file_path} does not exist.")
        return []
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

# Example usage
if __name__ == "__main__":
    # URL of the document
    url = 'https://docs.restech.unsw.edu.au/'

    # Save the data
    save_web_data(url)

    '''
    # Load the data
    corpus = load_data_chunks('web_data/web_data.json')

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
    '''
