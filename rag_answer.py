# rag.py

import os
from utils.config import (
    get_model_config,
    get_paths_config,
    get_system_config,
    AVAILABLE_MODELS,
    DEFAULT_MODEL
)
from utils import load_data_chunks
from utils.llm import create_client
from rag import create_standard_rag

def main():
    # Get configs
    paths = get_paths_config()
    system_config = get_system_config()
    hyce_config = system_config.get('hyce', {})
    
    # Use default model or specify one
    if not DEFAULT_MODEL:
        raise ValueError(f"No models available. Please check your config.yaml")
    
    # Create LLM client
    llm_client, model_name = create_client(DEFAULT_MODEL)
    
    # Load corpus
    data_path = os.path.join(paths.get('artifacts_dir', 'artifacts'), 
                            paths.get('web_data_file', 'web_data.json'))
    corpus = load_data_chunks(data_path)
    
    # Create RAG system with HyCE enabled
    rag = create_standard_rag(
        corpus=corpus,
        llm_client=llm_client,
        llm_model=model_name,
        use_hyce=hyce_config.get('enabled', True),
        commands_file=paths.get('commands_file', 'commands.json'),
        hyce_config=hyce_config
    )
    
    # Interactive loop
    print("\nWelcome to the RAG-based Q&A system!")
    print("This system can answer questions and execute relevant commands.")
    print("Type 'exit' to quit.")
    print("Add '--debug' after your question to see detailed information.\n")
    print(f"Using model: {model_name}")
    
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
            answer = rag.query(
                query=question,
                cot=True,
                debug=debug_mode
            )
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
