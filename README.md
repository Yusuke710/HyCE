# HyCE

## Overview
This project implements and evaluates a Retrieval-Augmented Generation (RAG) system with Hypothetical Command Embedding (HyCE). It combines web-scraped documentation with command-line utilities to provide accurate responses. The specific application including answering to HPC-related questions.

## Key Features
The project includes two main components:

1. **HyCE Implementation (`hyce.py`)**:
   - Core HyCE class-based implementation. It is just another RAG system with HyDE and execute the command as the corresponding command description is retrieved.
   - Pluggable to any RAG system
   - Includes debug mode
   - Configurable through `config.yaml`

2. **RAG with HyCE (`rag.py`)**:
   - Interactive Q&A system using HyCE
   - Supports command execution and web documentation
   - Terminal-based interface

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Project Structure
```
.
├── artifacts/                   # Generated data and results
│   ├── web_data.json           # Scraped web data
│   └── evaluation_dataset.csv  # Synthetic evaluation data
├── exp/                        # Experiments and evaluation
│   ├── rag_synthetic_data.py   # Test data generation
│   └── rag_eval.py            # Evaluation framework
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   └── llm.py                 # LLM interaction utilities
│   └── web_scraper.py         # Web scraping utilities  
├── hyce.py                    # HyCE implementation
├── rag.py                     # Abstract RAG system 
├── chatbot.py                 # Example RAG system with HyCE 
├── config.yaml                # System configuration
└── commands.json              # Command definitions
```

### Usage

**Interactive Q&A:**
```bash
export OPENAI_API_KEY=bashjdcbjasbcksdnbkcbdskbck
python chatbot.py
```

On first run, the system will:
1. Scrape documentation from UNSW's Katana HPC documentation
2. Create embeddings and cache them for future use
3. Start an interactive Q&A session

**Interactive Commands:**
- Type your question and press Enter
- Add `--debug` after your question to see detailed retrieval and processing information
- Type 'exit' or 'quit' to end the session

**Example Session:**
```bash
Question: What software does Katana have?
Answer: Katana provides various software packages including Python, R, and bioscience tools...

Question: How do I check my job status? --debug
=== Document Retrieval ===
Document retrieval took: 0.23 seconds
Found 5 documents...
[Debug information follows]

Answer: You can check your job status using the 'squeue' command...
```

### Configuration

The system is configured through `config.yaml`:

**Model Selection:**

1. **Via Config File:**
   - Set the default model in `config.yaml`:
   ```yaml
   models:
     default: gpt-4o-2024-08-06
   ```
  
Available model types:
- `openai`: OpenAI models (requires API key)
- `anthropic`: Anthropic Claude models (requires API key)
- `openrouter`: OpenRouter models (requires API key)
- `bedrock`: AWS Bedrock models (requires AWS credentials)
- `vertex_ai`: Google Vertex AI models (requires GCP credentials)
- `deepseek`: DeepSeek models (requires API key)
- `gemini`: Google Gemini models (requires API key)

**Embedding Options:**

1. **OpenAI Embeddings:**
   ```yaml
   system:
     embedding_type: "openai-ada"
   ```

2. **Sentence Transformers:**
   ```yaml
   system:
     embedding_type: "sentence-transformer"
   ```

**Cache Management:**
- Embeddings are cached in `artifacts/embeddings/`
- Document data is cached in `artifacts/web_data.json`
- Delete these files to embed new data


### FAISS compatibility issues with MPS

When running the system, you may encounter FAISS compatibility issues with MPS. To fix this, set the following environment variables:

```bash
export OMP_NUM_THREADS=1
export PYTORCH_MPS_DISABLE=1
export LLAMA_NO_METAL=1
```

Source: [https://neuml.github.io/txtai/faq/](https://neuml.github.io/txtai/faq/)

## Contributing
Feel free to submit issues and enhancement requests.

## License
See LICENSE.txt for details.


