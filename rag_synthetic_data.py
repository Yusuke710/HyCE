import os
import json
import random
import openai
import pandas as pd
from tqdm import tqdm
import re  # For regular expressions
import backoff  # For handling retries
import numpy as np
import torch

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set in the environment variable

# Your provided functions (without any modifications)
MAX_NUM_TOKENS = 500  # Adjust as needed

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.Timeout))
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    if model.startswith("claude-"):
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            max_completion_tokens=MAX_NUM_TOKENS,
            n=1,
            # stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history

def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found

# Load the data chunks from 'web_data/web_data.json'
def load_data_chunks(json_file_path):
    """
    Loads the web data from a JSON file.
    """
    if not os.path.exists(json_file_path):
        print(f"The file {json_file_path} does not exist.")
        return []
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data_chunks = json.load(json_file)
    return data_chunks

# Prepare source documents
def prepare_documents(data_chunks):
    """
    Converts data chunks into documents.
    """
    documents = []
    for chunk in data_chunks:
        documents.append({
            'text': chunk['chunk'],
            'source': chunk['url']
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
    system_message = ""
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

2. **Relevance**: The question is useful and relevant to users interested in the context.

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
    system_message = ""
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
    # Load data chunks
    data_chunks = load_data_chunks('web_data/web_data.json')

    # Prepare documents
    documents = prepare_documents(data_chunks)

    # Initialize outputs
    outputs = []

    # Number of QA pairs to generate
    N_GENERATIONS = 10  # Adjust as needed

    print(f"Generating {N_GENERATIONS} QA pairs...")

    num_docs = len(documents)

    # Randomly sample indices
    sampled_indices = random.sample(range(num_docs), min(N_GENERATIONS, num_docs))

    # Prepare the LLM client and model
    client = openai
    model = 'gpt-4o-2024-08-06'  # Using a supported model from your function

    for idx in tqdm(sampled_indices):
        # Get start and end indices for context
        start_idx = max(idx - 2, 0)
        end_idx = min(idx + 3, num_docs)  # +3 because end index is exclusive

        # Get the context documents
        context_docs = documents[start_idx:end_idx]

        # Concatenate their 'text' fields to form the context
        context = ' '.join([doc['text'] for doc in context_docs])

        # Source documents (for reference)
        source_docs = [doc['source'] for doc in context_docs]

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
    eval_dataset.to_csv("evaluation_dataset.csv", index=False)
    print("Synthetic evaluation dataset saved to 'evaluation_dataset.csv'")
