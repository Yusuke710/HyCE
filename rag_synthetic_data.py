import os
import random
import openai
import pandas as pd
from tqdm import tqdm

from rag.web_scrape_by_tag import load_data_chunks
from rag.llm import get_response_from_llm, extract_json_between_markers
from command_embedding_hyce import load_commands, get_command_output

# Prepare source documents
def prepare_documents(data_chunks):
    """
    Converts data chunks into documents.
    """
    documents = []
    for chunk in data_chunks:
        documents.append({
            'text': chunk['chunk'],
            'source': chunk.get('url', 'command')  # Use 'command' if 'url' is not present
        })
    return documents

# Function to call the LLM for generating QA pairs using get_response_from_llm
def generate_qa_pair(context, client, model):
    """
    Generates a question-answer pair based on the provided context.
    """
    # Modified QA_generation_prompt to output JSON
    QA_generation_prompt = f"""
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Imagine you are assisting a new user to High Performance Computing (HPC), and generate a question that a new user to HPC might ask.
Your factoid question should be formulated in the same style as questions new users to HPC would ask.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer in the following JSON format:

```json
{{
  "question": "Your generated question here",
  "answer": "Your answer to the question here"
}}
```

Now here is the context.

Context: {context}
"""
    system_message = "You are a helpful assistant that generates factoid questions and answers."
    try:
        # Call the LLM using the provided function
        response_text, _ = get_response_from_llm(
            msg=QA_generation_prompt,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=False
        )
        # Extract JSON from the LLM's response
        parsed_json = extract_json_between_markers(response_text)
        if parsed_json and "question" in parsed_json and "answer" in parsed_json:
            question = parsed_json["question"].strip()
            answer = parsed_json["answer"].strip()
            if len(answer) > 300:
                raise ValueError("Answer is too long")
            return question, answer
        else:
            raise ValueError("JSON parsing failed or missing keys")
    except Exception as e:
        print(f"Error generating QA pair: {e}")
        return None, None

# Function to call the LLM for combined critique using get_response_from_llm
def critique_question_combined(question, context, client, model):
    """
    Critiques the question based on groundedness, relevance, and standalone criteria.
    """
    critique_prompt = f"""
You will be given a context and a question.

Your task is to evaluate the question based on the following criteria:

1. **Groundedness**: The question can be answered unambiguously with the given context.

2. **Relevance**: The question addresses a common, practical problem or concept that HPC users are likely to encounter given the context.

3. **Standalone**: The question is understandable without additional context.

For each criterion, provide a 'total rating' of **1** (meets the criterion) or **0** (does not meet the criterion).

Provide your answer in the following JSON format:

```json
{{
  "evaluation": "Your rationale for the ratings",
  "groundedness_score": 0 or 1,
  "relevance_score": 0 or 1,
  "standalone_score": 0 or 1
}}
```

You MUST provide values for all keys in your answer.

Now here are the question and context.

Question: {question}

Context: {context}
"""
    system_message = "You are a helpful assistant that evaluates questions."
    try:
        # Call the LLM using the provided function
        response_text, _ = get_response_from_llm(
            msg=critique_prompt,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=False
        )
        # Extract JSON from the LLM's response
        parsed_json = extract_json_between_markers(response_text)
        if parsed_json and all(key in parsed_json for key in ["evaluation", "groundedness_score", "relevance_score", "standalone_score"]):
            groundedness_score = int(parsed_json["groundedness_score"])
            relevance_score = int(parsed_json["relevance_score"])
            standalone_score = int(parsed_json["standalone_score"])
            evaluation = parsed_json["evaluation"].strip()
            return {
                "groundedness_score": groundedness_score,
                "relevance_score": relevance_score,
                "standalone_score": standalone_score,
                "evaluation": evaluation
            }
        else:
            raise ValueError("JSON parsing failed or missing keys")
    except Exception as e:
        print(f"Error in critique: {e}")
        return None

# Main code to generate the synthetic dataset
if __name__ == "__main__":
    # Load data chunks from web data
    data_chunks = load_data_chunks(os.path.join("artifacts", "web_data.json"))

    # Load commands
    commands = load_commands('commands.json')

    # Convert commands into data chunks and extend data_chunks
    command_chunks = [{'chunk': explanation, 'url': 'command', 'command': cmd} for cmd, explanation in commands.items()]
    data_chunks.extend(command_chunks)

    # Prepare documents
    documents = prepare_documents(data_chunks)

    # Initialize outputs
    outputs = []

    # Number of QA pairs to generate
    N_GENERATIONS = 110  # Adjust as needed

    print(f"Generating {N_GENERATIONS} QA pairs...")

    num_docs = len(documents)
    num_command_chunks = len(command_chunks)

    # Calculate the starting index of command_chunks in data_chunks
    command_start_index = len(data_chunks) - num_command_chunks

    # Get indices of command_chunks
    command_indices = list(range(command_start_index, len(data_chunks)))

    # Randomly sample additional indices to meet N_GENERATIONS, ensuring no duplicates
    remaining_indices = list(set(range(num_docs)) - set(command_indices))
    additional_sampled_indices = random.sample(
        remaining_indices, max(0, min(N_GENERATIONS - len(command_indices), len(remaining_indices)))
    )

    # Combine command indices with additional sampled indices
    sampled_indices = command_indices + additional_sampled_indices

    print(f"Sampled {len(sampled_indices)} indices, ensuring all command_chunks are included.")

    # Prepare the LLM client and model
    client = openai
    model = 'gpt-4o-2024-08-06'  # Replace with your desired model

    for idx in tqdm(sampled_indices):
        # Get the context document
        context_doc = documents[idx]
        
        # Build context, executing command if necessary
        text = context_doc['text']
        context = text
        if context_doc['source'] == 'command':
            # Retrieve the command
            command = next((chunk['command'] for chunk in data_chunks if chunk['chunk'] == text), None)
            if command:
                # Execute the command and include its output
                command_output = get_command_output(command)
                context += f"\nCommand Output:\n{command_output}"

        # Source documents (for reference)
        source_docs = context_doc['source']

        # Generate QA pair
        question, answer = generate_qa_pair(context, client, model)

        if question and answer:
            # Critique the question
            critique_result = critique_question_combined(question, context, client, model)

            if critique_result:
                groundedness_score = critique_result["groundedness_score"]
                relevance_score = critique_result["relevance_score"]
                standalone_score = critique_result["standalone_score"]

                # Check if all scores are 1
                if all(score == 1 for score in [groundedness_score, relevance_score, standalone_score]):
                    # Add the QA pair and evaluations to outputs
                    outputs.append({
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "source_docs": source_docs,
                        "groundedness_score": groundedness_score,
                        "relevance_score": relevance_score,
                        "standalone_score": standalone_score,
                        "evaluation": critique_result["evaluation"]
                    })

    # Convert outputs to DataFrame
    generated_questions = pd.DataFrame(outputs)

    # Display evaluation dataset
    pd.set_option("display.max_colwidth", None)
    print("Evaluation dataset:")
    print(
        generated_questions[
            [
                "question",
                "answer",
                "groundedness_score",
                "relevance_score",
                "standalone_score",
            ]
        ]
    )

    # Save the evaluation dataset
    eval_dataset = generated_questions.reset_index(drop=True)
    eval_dataset.to_csv(os.path.join('artifacts', 'evaluation_dataset.csv'), index=False)
    print("Synthetic evaluation dataset saved to 'evaluation_dataset.csv'")
