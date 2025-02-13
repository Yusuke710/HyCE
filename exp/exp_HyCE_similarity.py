import json
import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np

# Initialize the cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# Load the CSV file
df = pd.read_csv('artifacts/evaluation_dataset.csv')  # Replace with your actual CSV file path

# Extract the questions
questions = df['question'].tolist()

# Load commands and their descriptions from 'commands.json'
with open('commands.json', 'r') as f:
    commands = json.load(f)

# Extract command names and descriptions
command_names = list(commands.keys())
command_descriptions = list(commands.values())

# Initialize lists to store the best similarity scores per question
best_name_similarities = []
best_desc_similarities = []

# Iterate over each question
for question in questions:
    # Prepare pairs for command names
    name_pairs = [(question, name) for name in command_names]
    # Prepare pairs for command descriptions
    desc_pairs = [(question, desc) for desc in command_descriptions]

    # Compute similarity scores
    name_scores = cross_encoder.predict(name_pairs)
    desc_scores = cross_encoder.predict(desc_pairs)

    # Find the maximum similarity score (best match) for command names
    max_name_similarity = np.max(name_scores)
    best_name_index = np.argmax(name_scores)
    best_command = command_names[best_name_index]

    # Find the maximum similarity score (best match) for command descriptions
    max_desc_similarity = np.max(desc_scores)
    best_desc_index = np.argmax(desc_scores)
    best_desc = command_descriptions[best_desc_index]

    # Append the maximum similarities to the lists
    best_name_similarities.append(max_name_similarity)
    best_desc_similarities.append(max_desc_similarity)

    # Output the best match for this question (optional)
    print(f"Question: {question}")
    print(f"Best Matching Command Name: {best_command} (Similarity: {max_name_similarity:.4f})")
    print(f"Best Matching Command Description: {best_desc} (Similarity: {max_desc_similarity:.4f})")
    print("-" * 60)

# Calculate the overall average of the best similarities
overall_avg_best_name_similarity = np.mean(best_name_similarities)
overall_avg_best_desc_similarity = np.mean(best_desc_similarities)

# Output the overall average similarities
print("Overall Average Similarity of Best Matches with Command Names: {:.4f}".format(overall_avg_best_name_similarity))
print("Overall Average Similarity of Best Matches with Command Descriptions: {:.4f}".format(overall_avg_best_desc_similarity))
