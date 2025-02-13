# rag.py

import os
import openai
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.web_scrape_by_tag import load_data_chunks
from rag.text_embedding import (
    embed_corpus,
    save_embeddings,
    load_embeddings,
    build_faiss_index,
)
from rag.llm import get_response_from_llm, extract_json_between_markers
from command_embedding_hyce import load_commands, get_command_output
from rag.hyce import HyCE

MAX_NUM_TOKENS = 128000

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

    _, indices = search_with_faiss(query_embedding, index, 20)

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

def generate_rag_answer(question, corpus, corpus_embeddings, bi_encoder, cross_encoder, 
                       index, client, model, cot=False, hyce_enabled=False):
    """
    Generates an answer using the RAG system with optional HyCE support.
    """
    # First get relevant documents using existing search
    doc_chunks = search(
        question,
        corpus,
        corpus_embeddings,
        bi_encoder,
        cross_encoder,
        index,
        top_k=5
    )

    # Process with HyCE if enabled
    if hyce_enabled:
        hyce = HyCE(
            embedding_model=bi_encoder,
            cross_encoder=cross_encoder,
            similarity_threshold=0.5
        )
        # This will now do parallel retrieval, reranking, and command execution
        top_chunks = hyce.process_contexts(
            query=question,
            doc_contexts=doc_chunks,
            top_k_docs=5,
            top_k_commands=3,
            final_top_k=5
        )
    else:
        top_chunks = doc_chunks

    # Format context string
    context = ""
    for i, chunk_info in enumerate(top_chunks):
        if chunk_info.get('url') == 'command':
            context += f"Command {i+1} Explanation:\n{chunk_info['chunk']}\n"
            if 'command_output' in chunk_info:
                context += f"Command Output:\n{chunk_info['command_output']}\n"
            context += "\n"
        else:
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
        client=client,
        model=model,
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
    
    # Initialize models first
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    # Then initialize HyCE with the models
    hyce = HyCE(
        embedding_model=bi_encoder,
        cross_encoder=cross_encoder,
        commands_file='commands.json'
    )
    command_corpus = hyce.get_command_contexts()
    corpus.extend(command_corpus)

    if corpus:
        embeddings_file = os.path.join('artifacts', 'combined_corpus_embeddings.npy')
        corpus_embeddings = load_embeddings(embeddings_file)

        if corpus_embeddings is not None and len(corpus_embeddings) != len(corpus):
            corpus_embeddings = None
        
        if corpus_embeddings is None:
            print("No embeddings found, embedding corpus...")
            corpus_embeddings = embed_corpus(corpus, bi_encoder)
            save_embeddings(corpus_embeddings, embeddings_file)
            assert len(corpus_embeddings) == len(corpus), "Mismatch between corpus and corpus_embeddings lengths"

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
                model,
                cot=True,
                hyce_enabled=True
            )

            print("\nAnswer:")
            print(generated_answer)
    else:
        print("No data to process.")
