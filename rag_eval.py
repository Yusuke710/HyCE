import os
import json
import pandas as pd
import numpy as np
import openai  # Importing openai for the LLM client
from tqdm import tqdm

from main import *

# Define the functions for generating RAG answers and evaluating them
def generate_rag_answer(question, corpus, corpus_embeddings, bi_encoder, cross_encoder, index, client, model):
    # Retrieve relevant documents using the provided search function
    top_chunks = search(
        question,
        corpus,
        corpus_embeddings,
        bi_encoder,
        cross_encoder,
        index,
        top_k=5
    )

    # Prepare the context from top_chunks
    context = ""
    for i, chunk_info in enumerate(top_chunks):
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

    # Return the generated answer
    return content

def evaluate_answer(question, generated_answer, true_answer, client, model):
    evaluation_prompt = f"""
You will be given a question, a generated answer, and a reference answer.

Your task is to evaluate the generated answer based on the following criteria:

1. **Correctness**: The answer is correct and accurate based on the reference answer.

2. **Faithfulness**: The answer does not hallucinate or contradict any factual information presented in the context.

3. **Summarization Quality**: The answer effectively summarizes the context to provide a concise response.

For each criterion, provide a 'total rating' of **1** (meets the criterion) or **0** (does not meet the criterion).

Output your reasoning for the ratings in the evaluation field and provide your scores in the following JSON format:

```json
{{
  "evaluation": "Your detailed feedback explaining the evaluation based on the criteria.",
  "scores": {{
    "Correctness": 0 or 1,
    "Faithfulness": 0 or 1,
    "Summarization Quality": 0 or 1
  }}
}}
Now, here are the inputs:

Question: {question}

Generated Answer: {generated_answer}

Reference Answer: {true_answer} """

    system_message = "You are a fair evaluator language model."
    try:
        # Call the LLM using the provided function
        response_text, _ = get_response_from_llm(
            msg=evaluation_prompt,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=False
        )
        # Extract JSON from the LLM's response
        parsed_json = extract_json_between_markers(response_text)
        if parsed_json and "evaluation" in parsed_json and "scores" in parsed_json:
            evaluation = parsed_json["evaluation"].strip()
            scores = parsed_json["scores"]
            # Ensure that scores are integers
            scores = {k: int(v) for k, v in scores.items()}
            return {
                "evaluation": evaluation,
                "scores": scores
            }
        else:
            raise ValueError("JSON parsing failed or missing keys in evaluation.")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None

# Main evaluation code
if __name__ == "__main__":
    # Load the synthetic evaluation dataset
    eval_dataset = pd.read_csv("evaluation_dataset.csv")

    # Load the corpus
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

        # Prepare the LLM client and model
        openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure API key is set

        client = openai
        model = 'gpt-4o-2024-08-06'  # Using a supported model from your function

        # Initialize results list
        results = []

        print("Running evaluation...")

        for idx, row in tqdm(eval_dataset.iterrows(), total=eval_dataset.shape[0]):
            question = row['question']
            true_answer = row['answer']

            # Generate an answer using the RAG system
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

            # Evaluate the generated answer
            evaluation_result = evaluate_answer(
                question,
                generated_answer,
                true_answer,
                client,
                model
            )

            if evaluation_result:
                # Extract individual scores
                correctness_score = evaluation_result['scores'].get('Correctness', None)
                faithfulness_score = evaluation_result['scores'].get('Faithfulness', None)
                summarization_score = evaluation_result['scores'].get('Summarization Quality', None)

                results.append({
                    'question': question,
                    'true_answer': true_answer,
                    'generated_answer': generated_answer,
                    'evaluation': evaluation_result['evaluation'],
                    'Correctness': correctness_score,
                    'Faithfulness': faithfulness_score,
                    'Summarization Quality': summarization_score
                })

        # Save results to a JSON file
        with open('rag_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        # Load results into a DataFrame for analysis
        results_df = pd.DataFrame(results)

        # Display individual results
        print("\nIndividual Results:")
        print(results_df[['question', 'true_answer', 'generated_answer', 'Correctness', 'Faithfulness', 'Summarization Quality']])

        # Calculate and display average scores
        average_correctness = results_df['Correctness'].mean()
        average_faithfulness = results_df['Faithfulness'].mean()
        average_summarization = results_df['Summarization Quality'].mean()

        print(f"\nAverage Scores:")
        print(f"Correctness: {average_correctness:.2f} out of 1")
        print(f"Faithfulness: {average_faithfulness:.2f} out of 1")
        print(f"Summarization Quality: {average_summarization:.2f} out of 1")

        # Count answers where all three criteria are scored as 1
        low_score_count = results_df[
            (results_df['Correctness'] == 1) & 
            (results_df['Faithfulness'] == 1) & 
            (results_df['Summarization Quality'] == 1)
        ].shape[0]
        
        # Calculate the percentage
        total_count = results_df.shape[0]
        low_score_percentage = (low_score_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nNumber of answers with all scores as 1: {low_score_count}")
        print(f"Percentage of such answers: {low_score_percentage:.2f}%")
    else:
        print("No data to process.")
