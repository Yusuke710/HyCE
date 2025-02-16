import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
import pandas as pd
import openai
from tqdm import tqdm
from utils.config import get_system_config, get_paths_config
from utils import load_data_chunks, get_response_from_llm, extract_json_between_markers
from rag import create_standard_rag

def evaluate_answer(question, generated_answer, true_answer, client, model):
    evaluation_prompt = f"""
You will be given a question, a generated answer, and a reference answer.

Your task is to evaluate the generated answer based on the following criteria:

1. **Correctness**: The answer is correct and accurate based on the reference answer.

2. **Faithfulness**: The answer does not hallucinate or contradict any factual information presented in the context.

For each criterion, provide a 'total rating' of **1** (meets the criterion) or **0** (does not meet the criterion).

Output your reasoning for the ratings in the evaluation field and provide your scores in the following JSON format:

```json
{{
  "evaluation": "Your detailed feedback explaining the evaluation based on the criteria.",
  "scores": {{
    "Correctness": 0 or 1,
    "Faithfulness": 0 or 1,
  }}
}}
```

Now, here are the inputs:

Question: {question}

Generated Answer: {generated_answer}

Reference Answer: {true_answer}
"""

    system_message = "You are a helpful assistant that evaluates answers."
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
    # Load configs
    paths = get_paths_config()
    system_config = get_system_config()
    
    # Load evaluation dataset
    eval_dataset = pd.read_csv(os.path.join(paths.get('artifacts_dir', 'artifacts'), 
                                          'evaluation_dataset.csv'))
    
    # Load corpus
    data_path = os.path.join(paths.get('artifacts_dir', 'artifacts'), 
                            paths.get('web_data_file', 'web_data.json'))
    corpus = load_data_chunks(data_path)
    
    # Create RAG system
    client_answer = openai.OpenAI()
    model_answer = 'gpt-4o-2024-08-06'  # Replace with your desired model
    
    rag = create_standard_rag(
        corpus=corpus,
        llm_client=client_answer,
        llm_model=model_answer,
        embedding_type=system_config.get('embedding_type', 'sentence-transformer'),
        use_hyce=True,
        commands_file=paths.get('commands_file', 'commands.json')
    )
    
    # Prepare evaluation client
    client_eval = openai.OpenAI()
    model_eval = 'gpt-4o-2024-08-06'
    
    # Initialize results
    results = []
    
    print("Running evaluation...")
    
    for idx, row in tqdm(eval_dataset.iterrows(), total=eval_dataset.shape[0]):
        question = row['question']
        true_answer = row['answer']
        context = row['context']
        
        # Generate answer using RAG
        generated_answer = rag.query(
            query=question,
            cot=True,
            debug=False
        )
        
        # Evaluate answer
        evaluation_result = evaluate_answer(
            question,
            generated_answer,
            true_answer,
            client_eval,
            model_eval
        )
        
        if evaluation_result:
            # Extract individual scores
            correctness_score = evaluation_result['scores'].get('Correctness', None)
            faithfulness_score = evaluation_result['scores'].get('Faithfulness', None)

            results.append({
                'context': context,
                'question': question,
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'evaluation': evaluation_result['evaluation'],
                'Correctness': correctness_score,
                'Faithfulness': faithfulness_score,
            })

    # Save results to a JSON file
    with open(os.path.join('artifacts', 'rag_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Load results into a DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Display individual results
    print("\nIndividual Results:")
    print(results_df[['context', 'question', 'true_answer', 'generated_answer', 'Correctness', 'Faithfulness']])

    # Calculate and display average scores
    average_correctness = results_df['Correctness'].mean()
    average_faithfulness = results_df['Faithfulness'].mean()

    print(f"\nAverage Scores:")
    print(f"Correctness: {average_correctness:.2f} out of 1")
    print(f"Faithfulness: {average_faithfulness:.2f} out of 1")

    # Count answers where all three criteria are scored as 1
    high_score_count = results_df[
        (results_df['Correctness'] == 1) &
        (results_df['Faithfulness'] == 1) 
    ].shape[0]

    # Calculate the percentage
    total_count = results_df.shape[0]
    high_score_percentage = (high_score_count / total_count) * 100 if total_count > 0 else 0

    print(f"\nNumber of answers with all scores as 1: {high_score_count}")
    print(f"Percentage of such answers: {high_score_percentage:.2f}%")
