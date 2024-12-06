# command_hyde_with_embedding_refactored.py

import os
import json
from sentence_transformers import SentenceTransformer
import subprocess
import faiss
import torch

# Import reusable functions from text_embedding.py
from rag.text_embedding import save_embeddings, load_embeddings, build_faiss_index
from rag.llm import create_client

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

def retrieve_explanation(query, explanations, index, client_embed, model_embed):
    """
    Retrieves the most similar explanation to the query.
    """
    # Embed the query as a PyTorch tensor
    query_embedding = get_embedding(query, client_embed, model_embed)
    # Convert to NumPy array for FAISS and reshape to 2D array
    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
    # Normalize query embedding
    faiss.normalize_L2(query_embedding_np)
    distances, indices = index.search(query_embedding_np, k=1)
    closest_idx = indices[0][0]
    command_name = list(explanations.keys())[closest_idx]
    explanation = explanations[command_name]
    return command_name, explanation

def generate_hyde_context(query, explanations, index, client_embed, model_embed):
    """
    Retrieves the relevant command explanation based on the query,
    runs the command if explanation is retrieved, and returns both as context.
    """
    command_name, explanation = retrieve_explanation(query, explanations, index, client_embed, model_embed)
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
    client_embed, model_embed = create_client("llama-3.2-nv-embedqa-1b-v1")

    # Load commands and explanations from the JSON file
    commands_file = 'commands.json'
    COMMAND_EXPLANATIONS = load_commands(commands_file)

    embeddings_file = os.path.join('artifacts', 'command_explanations_embeddings.npy')

    # Step 1: Embed explanations and save to a file if not already done
    if os.path.exists(embeddings_file):
        embeddings = load_embeddings(embeddings_file)
    else:
        embeddings = []
        for cmd, explanation in COMMAND_EXPLANATIONS.items():
            embedding = get_embedding(explanation, client_embed, model_embed)
            embeddings.append(embedding)
        save_embeddings(embeddings, embeddings_file)

    # Step 2: Build the FAISS index
    embeddings_np = embeddings.cpu().numpy()
    faiss_index = build_faiss_index(embeddings)

    # Step 3: Query and retrieve context
    user_query = "What is my username?"
    context = generate_hyde_context(user_query, COMMAND_EXPLANATIONS, faiss_index, client_embed, model_embed)
    print(json.dumps(context, indent=4))
