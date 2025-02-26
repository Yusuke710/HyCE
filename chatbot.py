# rag.py

import os
import argparse
import json
import datetime
from utils.config import (
    get_paths_config,
    get_system_config,
    get_model_config,
    get_embedding_config,
    get_reranker_config
)
from utils import load_data_chunks
from utils.llm import create_client
from utils.web_scraper import scrape_documentation
from rag import create_standard_rag

def ensure_data_exists(paths):
    """Ensure the web data exists, if not, run the scraper."""
    data_path = os.path.join(paths.get('artifacts_dir', 'artifacts'), 
                            paths.get('web_data_file', 'web_data.json'))
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    if not os.path.exists(data_path):
        print("\nNo existing documentation found. Starting web scraping...")
        try:
            scrape_documentation(output_path=data_path)
            print("Web scraping completed successfully!")
        except Exception as e:
            raise Exception(f"Failed to scrape documentation: {str(e)}")
    
    return data_path

def parse_arguments():
    """Parse command line arguments."""
    # Get current system username as default
    current_user = os.environ.get('USER', '')
    
    parser = argparse.ArgumentParser(description='RAG-based Q&A system')
    parser.add_argument('-u', '--username', default=current_user, 
                        help=f'Specify the username (default: current user "{current_user}")')
    parser.add_argument('-m', '--model', help='Specify the LLM model to use')
    parser.add_argument('-e', '--embedding', help='Specify the embedding model to use')
    parser.add_argument('-r', '--reranker', help='Specify the reranker model to use')
    return parser.parse_args()

def save_feedback(query, retrieved_docs, answer, feedback, paths):
    """Save user feedback along with query, retrieved docs, and answer."""
    # Get artifacts directory from paths config
    feedback_dir = os.path.join(paths.get('artifacts_dir', 'artifacts'))
    
    feedback_path = os.path.join(feedback_dir, paths.get('feedback_file', 'feedback_data.jsonl'))
    
    # Create feedback entry
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "retrieved_docs": retrieved_docs,
        "answer": answer,
        "feedback": feedback,
    }
    
    # Append to JSONL file
    with open(feedback_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return feedback_path

def get_user_feedback():
    """Get user feedback (y/n) with validation."""
    while True:
        feedback = input("Was this answer helpful? (y/n): ").strip().lower()
        if feedback in ['y', 'n']:
            return feedback
        print("Please enter 'y' for yes or 'n' for no.")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set username as environment variable (this will either be the provided username
    # or the current user that was already set as default)
    os.environ['USER'] = args.username
    
    # Get configs
    paths = get_paths_config()
    system_config = get_system_config()
    hyce_config = system_config.get('hyce', {})
    
    # Get model configurations
    default_model = system_config.get('default_model')
    default_embedding = system_config.get('default_embedding')
    default_reranker = system_config.get('default_reranker')
    
    # Use command line arguments if provided, otherwise use defaults
    model_name = args.model or default_model
    embedding_name = args.embedding or default_embedding
    reranker_name = args.reranker or default_reranker
    
    if not model_name:
        raise ValueError("No model specified and no default model in config")
    
    print("Initializing the system...")
    print(f"User: {args.username}")
    print(f"LLM Model: {model_name}")
    print(f"Embedding Model: {embedding_name}")
    print(f"Reranker Model: {reranker_name}")
    
    # Ensure we have the required data
    try:
        data_path = ensure_data_exists(paths)
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return
    
    # Create LLM client
    llm_client, model_name = create_client(model_name)
    
    # Load corpus
    corpus = load_data_chunks(data_path)
    
    # Get embedding and reranker configurations
    embedding_config = get_embedding_config(embedding_name)
    reranker_config = get_reranker_config(reranker_name)
    
    # Create RAG system with HyCE enabled
    rag = create_standard_rag(
        corpus=corpus,
        llm_client=llm_client,
        llm_model=model_name,
        embedding_type=embedding_name,
        use_hyce=hyce_config.get('enabled', True),
        commands_file=paths.get('commands_file', 'commands.json'),
        hyce_config=hyce_config
    )
    
    # Interactive loop
    print("\nWelcome to the RAG-based Q&A system!")
    print("This system can answer questions and execute relevant commands.")
    print("Type 'exit' to quit.")
    print("Add '--debug' after your question to see detailed information.\n")
    
    while True:
        # Get user input
        question = input("\nQuestion: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        
        # Check for debug mode
        debug_mode = False
        if question.endswith('--debug'):
            debug_mode = True
            question = question.replace('--debug', '').strip()
        
        # Get answer with chain-of-thought reasoning
        try:
            # Store the query
            query = question
            
            # First, get the retrieved documents using the RAG's retrieve method
            retrieved_docs = []
            try:
                # Use the RAG's retrieve method directly
                if hasattr(rag, 'retrieve'):
                    contexts = rag.retrieve(query, top_k=5, debug=debug_mode)
                    
                    # Format the retrieved documents for storage
                    for ctx in contexts:
                        doc_entry = {}
                        
                        # Extract content based on type
                        if ctx.get('url') == 'command':
                            doc_entry['type'] = 'command'
                            doc_entry['command'] = ctx.get('command', '')
                            doc_entry['description'] = ctx.get('chunk', '')
                            if 'command_output' in ctx:
                                doc_entry['output'] = ctx.get('command_output', '')
                        else:
                            doc_entry['type'] = 'document'
                            doc_entry['content'] = ctx.get('chunk', '')[:500] + "..." if len(ctx.get('chunk', '')) > 500 else ctx.get('chunk', '')
                            doc_entry['url'] = ctx.get('url', '')
                        
                        # Add scores if available
                        if 'score' in ctx:
                            doc_entry['score'] = ctx.get('score')
                        if 'cross_encoder_score' in ctx:
                            doc_entry['rerank_score'] = ctx.get('cross_encoder_score')
                        
                        retrieved_docs.append(doc_entry)
            except Exception as retrieval_error:
                if debug_mode:
                    print(f"Could not retrieve documents: {str(retrieval_error)}")
            
            # Get answer
            answer = rag.query(
                query=question,
                cot=True,
                debug=debug_mode
            )
            
            print("\nAnswer:", answer)
            
            # Get user feedback
            feedback = get_user_feedback()
            
            # Save feedback data
            feedback_path = save_feedback(query, retrieved_docs, answer, feedback, paths)
            if debug_mode:
                print(f"Feedback saved to {feedback_path}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
