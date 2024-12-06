
import os
import openai
import requests
import faiss
import torch
import numpy as np

from rag.web_scrape_by_tag import load_data_chunks
from rag.text_embedding import (
    embed_corpus,
    save_embeddings,
    load_embeddings,
    build_faiss_index,
)
from rag.llm import get_response_from_llm, extract_json_between_markers, create_client
from command_embedding_hyde import load_commands, get_command_output

MAX_NUM_TOKENS = 128000

def get_embedding(text, client, model):
    """
    Gets the embedding of a single text input using the NVIDIA embedding model via NVIDIA API.
    """
    print(f"Getting embedding for: {text}")
    response = client.embeddings.create(
    input=[text],
    model=model,
    encoding_format="float",
    extra_body={"input_type": "query", "truncate": "END"}
)

    embedding = response.data[0].embedding
    return torch.tensor(embedding)

def search_with_faiss(query_embedding, index, top_k):
    """
    Searches the FAISS index for the top_k most similar embeddings.
    """
    query_embedding = np.array([query_embedding.numpy()])  # Make it a 2D array
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

def rerank_passages(query, passages):
    """
    Calls the NVIDIA rerank model to rerank the passages given the query.
    Returns the rankings as a list of {'index': idx, 'logit': score}.
    """
    invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v1/reranking"
    headers = {
        "Authorization": "Bearer your_api_key",
        "Accept": "application/json",
    }
    payload = {
        "model": "nvidia/llama-3.2-nv-rerankqa-1b-v1",
        "query": {
            "text": query
        },
        "passages": passages
    }

    # Use a session for connection pooling
    session = requests.Session()

    response = session.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()

    # The response contains 'rankings', which is a list of {'index': idx, 'logit': score}
    rankings = response_body.get('rankings', [])

    # Sort the rankings in descending order of logit
    rankings = sorted(rankings, key=lambda x: x['logit'], reverse=True)

    return rankings

def search(query, corpus, index, client, model, top_k=5):
    """
    Searches for the most relevant chunks to the query, using FAISS and NVIDIA rerank model.
    """
    # Get the query embedding
    query_embedding = get_embedding(query, client, model)

    # Search with FAISS
    distances, indices = search_with_faiss(query_embedding, index, top_k=20)

    # Collect the passages and their indices
    passages = []
    passage_indices = []
    for idx in indices:
        idx = int(idx)
        passage_indices.append(idx)
        chunk_info = corpus[idx]
        passages.append({"text": chunk_info['chunk'][:1380]}) # trancate up to 1400 characters ~ 466 tokens

    # Rerank the passages using NVIDIA rerank model
    rankings = rerank_passages(query, passages)

    # Rearrange the passages according to the reranked indices
    top_chunks = []
    for rank_info in rankings:
        idx = rank_info['index']
        score = rank_info['logit']
        corpus_idx = passage_indices[idx]
        chunk_info = corpus[corpus_idx].copy()
        chunk_info['score'] = float(score)
        top_chunks.append(chunk_info)

    # Limit to top_k
    top_chunks = top_chunks[:top_k]

    return top_chunks

def generate_rag_answer(question, corpus, index, client_embed, client_answer, model_embed, model_answer, cot=False, HyCE=False):
    """
    Generates an answer using the RAG system.
    """
    # Retrieve relevant documents
    top_chunks = search(
        question,
        corpus,
        index,
        client_embed,
        model_embed, 
        top_k=5
    )

    # Prepare the context from top_chunks, including command execution if applicable
    context = ""
    if HyCE:
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
    else:
        for i, chunk_info in enumerate(top_chunks):
            context += f"Document {i+1}:\n{chunk_info['chunk']}\n\n"

    # Limit context length to avoid token limits
    max_chars = 4*MAX_NUM_TOKENS  # Approximate limit that works well with most models
    if len(context) > max_chars:
        context = context[:max_chars]
    
    # Prepare the system message and user message
    system_message = "You are an assistant that provides accurate and concise answers based on the provided documents."

    if cot:
        user_message = f"""Answer the following question based on the provided documents.
        Question: {question}

        Documents:
        {context} 

        Provide your thinking for each step, and then provide the answer in the following **JSON format**:

        ```json
        {{
            "reasons": <reasons>,
            "answer": <answer>
        }}
        ```
        """
    else:
        user_message = f"Answer the following question based on the provided documents.\n\nQuestion: {question}\n\nDocuments:\n{context}"

    # Call the LLM using the provided function
    content, _ = get_response_from_llm(
        msg=user_message,
        client=client_answer,
        model=model_answer,
        system_message=system_message,
        print_debug=False
    )

    if cot:
        # Parse the JSON response to extract the path
        response_json = extract_json_between_markers(content)
        if response_json is None:
            return None
        answer = response_json.get("answer", [])
        return answer
    else:
        return content

# Main code
if __name__ == "__main__":
    corpus = load_data_chunks('artifacts/web_data.json')
    commands = load_commands('commands.json')

    client_embed, model_embed = create_client("llama-3.2-nv-embedqa-1b-v1")
    client_answer, model_answer  = create_client("gpt-4o-2024-08-06")

    # Convert commands to corpus format and append them to the corpus
    command_corpus = [{'chunk': explanation, 'url': 'command', 'command': cmd} for cmd, explanation in commands.items()]
    corpus.extend(command_corpus)

    if corpus:
        embeddings_file = os.path.join('artifacts', 'combined_corpus_embeddings.npy')
        corpus_embeddings = load_embeddings(embeddings_file)

        if corpus_embeddings is not None and len(corpus_embeddings) != len(corpus):
            corpus_embeddings = None
        

        if corpus_embeddings is None:
            print("No embeddings found, embedding corpus...")
            corpus_embeddings = embed_corpus(corpus, client_embed, model_embed)
            save_embeddings(corpus_embeddings, embeddings_file)
            assert len(corpus_embeddings) == len(corpus), "Mismatch between corpus and corpus_embeddings lengths"

        index = build_faiss_index(corpus_embeddings)

        print("Welcome to the RAG-based Q&A system. Type 'exit' to quit.")
        while True:
            question = input("\nPlease enter your question: ")
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            generated_answer = generate_rag_answer(
                question,
                corpus,
                index,
                client_embed,
                client_answer,
                model_embed,
                model_answer,
                cot=True
            )

            print("\nAnswer:")
            print(generated_answer)
    else:
        print("No data to process.")
