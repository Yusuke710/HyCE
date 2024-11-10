# command_hyde_with_embedding_refactored.py

import os
import json
from sentence_transformers import SentenceTransformer
import subprocess
import faiss

# Import reusable functions from text_embedding.py
from rag.text_embedding import save_embeddings, load_embeddings, build_faiss_index

# Load commands and explanations from JSON file
def load_commands(file_path='commands.json'):
    """
    Loads commands and their explanations from a JSON file.
    """
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return {}
    with open(file_path, 'r', encoding='utf-8') as file:
        commands = json.load(file)
    return commands

def get_command_output(command, timeout=10):
    """
    Executes a shell command and returns its output as a string, with a timeout.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command '{command}' timed out after {timeout} seconds.")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        return ""

def embed_explanations(explanations, model):
    """
    Embeds the command explanations and returns the embeddings as a PyTorch tensor.
    """
    explanation_texts = list(explanations.values())
    embeddings = model.encode(
        explanation_texts,
        batch_size=4,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return embeddings

def retrieve_explanation(query, explanations, index, model):
    """
    Retrieves the most similar explanation to the query.
    """
    # Embed the query as a PyTorch tensor
    query_embedding = model.encode([query], convert_to_tensor=True)
    # Convert to NumPy array for FAISS
    query_embedding_np = query_embedding.cpu().numpy()
    faiss.normalize_L2(query_embedding_np)  # Normalize query embedding
    distances, indices = index.search(query_embedding_np, k=1)
    closest_idx = indices[0][0]
    command_name = list(explanations.keys())[closest_idx]
    explanation = explanations[command_name]
    return command_name, explanation

def generate_hyde_context(query, explanations, index, model):
    """
    Retrieves the relevant command explanation based on the query,
    runs the command if explanation is retrieved, and returns both as context.
    """
    command_name, explanation = retrieve_explanation(query, explanations, index, model)
    output = get_command_output(command_name)
    context = {
        'command': command_name,
        'explanation': explanation,
        'output': output
    }
    return context

# Example usage
if __name__ == '__main__':
    # Initialize the bi-encoder model for embeddings
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model

    # Load commands and explanations from the JSON file
    commands_file = 'commands.json'
    COMMAND_EXPLANATIONS = load_commands(commands_file)

    embeddings_file = os.path.join('artifacts', 'command_explanations_embeddings.npy')

    # Step 1: Embed explanations and save to a file if not already done
    if os.path.exists(embeddings_file):
        embeddings = load_embeddings(embeddings_file)
    else:
        embeddings = embed_explanations(COMMAND_EXPLANATIONS, bi_encoder)
        save_embeddings(embeddings, embeddings_file)

    # Step 2: Build the FAISS index
    embeddings_np = embeddings.cpu().numpy()
    faiss_index = build_faiss_index(embeddings)

    # Step 3: Query and retrieve context
    user_query = "What is my username?"
    context = generate_hyde_context(user_query, COMMAND_EXPLANATIONS, faiss_index, bi_encoder)
    print(json.dumps(context, indent=4))
