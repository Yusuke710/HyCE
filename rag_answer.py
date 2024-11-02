# rag.py

import os
import json
import torch
import numpy as np
import openai
import faiss
import subprocess
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.web_scrape import load_data_chunks
from rag.text_embedding import (
    embed_corpus,
    save_embeddings,
    load_embeddings,
    build_faiss_index,
)
from rag.llm import get_response_from_llm
from rag.command_embedding_hyde import load_commands, get_command_output

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
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.unsqueeze(0)  # Add batch dimension

    distances, indices = search_with_faiss(query_embedding, index, 100)

    cross_inp = []
    cross_chunks = []
    for idx in indices:
        idx = int(idx)
        chunk = corpus[idx]['chunk']
        cross_inp.append((query, chunk))
        chunk_info = corpus[idx].copy()
        cross_chunks.append(chunk_info)

    cross_scores = cross_encoder.predict(cross_inp)

    for i, score in enumerate(cross_scores):
        cross_chunks[i]['score'] = score

    cross_chunks = sorted(cross_chunks, key=lambda x: x['score'], reverse=True)

    seen = set()
    unique_chunks = []
    for chunk in cross_chunks:
        identifier = (chunk.get('url'), chunk['chunk'])
        if identifier not in seen:
            seen.add(identifier)
            unique_chunks.append(chunk)
        if len(unique_chunks) >= top_k:
            break

    return unique_chunks

def generate_rag_answer(question, corpus, corpus_embeddings, bi_encoder, cross_encoder, index, client, model):
    """
    Generates an answer using the RAG system.
    """
    # Retrieve relevant documents
    top_chunks = search(
        question,
        corpus,
        corpus_embeddings,
        bi_encoder,
        cross_encoder,
        index,
        top_k=5
    )

    # Prepare the context from top_chunks, including command execution if applicable
    context = ""
    for i, chunk_info in enumerate(top_chunks):
        # If the chunk is a command explanation, execute the command
        if chunk_info.get('url') == 'command':
            command = chunk_info['command']
            output = get_command_output(command)
            context += f"Command {i+1} Explanation:\n{chunk_info['chunk']}\n"
            context += f"Command Output:\n{output}\n\n"
        else:
            # Otherwise, add the web-scraped content
            context += f"Document {i+1}:\n{chunk_info['chunk']}\n\n"

    # Prepare the system message and user message
    system_message = "You are an assistant that provides accurate and concise answers based on the provided documents."
    user_message = f"Answer the following question based on the provided documents.\n\nQuestion: {question}\n\nDocuments:\n{context}"

    # Call the LLM using the provided function
    content, _ = get_response_from_llm(
        msg=user_message,
        client=client,
        model=model,
        system_message=system_message,
        print_debug=False
    )

    return content

# Main code
if __name__ == "__main__":
    corpus = load_data_chunks('web_data/web_data.json')
    commands = load_commands('commands.json')

    # Convert commands to corpus format and append them to the corpus
    command_corpus = [{'chunk': explanation, 'url': 'command', 'command': cmd} for cmd, explanation in commands.items()]
    corpus.extend(command_corpus)

    if corpus:
        bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

        embeddings_file = 'combined_corpus_embeddings.npy'
        corpus_embeddings = load_embeddings(embeddings_file)

        if corpus_embeddings is None:
            corpus_embeddings = embed_corpus(corpus, bi_encoder)
            save_embeddings(corpus_embeddings, embeddings_file)

        index = build_faiss_index(corpus_embeddings)

        openai.api_key = os.getenv('OPENAI_API_KEY')

        client = openai
        model = 'gpt-4o-2024-08-06'

        print("Welcome to the RAG-based Q&A system. Type 'exit' to quit.")
        while True:
            question = input("\nPlease enter your question: ")
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            generated_answer = generate_rag_answer(
                question,
                corpus,
                corpus_embeddings,
                bi_encoder,
                cross_encoder,
                index,
                client,
                model
            )

            print("\nAnswer:")
            print(generated_answer)
    else:
        print("No data to process.")
