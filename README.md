# HyCE

## Overview
This project implements and evaluates a Retrieval-Augmented Generation (RAG) system with Hypothetical Command Embedding (HyCE). It combines web-scraped documentation with command-line utilities to provide accurate responses. The specific application including answering to HPC-related questions.

## Key Features
The project includes two main components:

1. **HyCE Implementation (`hyce.py`)**:
   - Core HyCE class-based implementation
   - Pluggable to any RAG system
   - Includes debug mode
   - Configurable through `config.yaml`

2. **RAG with HyCE (`rag_answer.py`)**:
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
├── hyce.py                    # HyCE implementation
├── rag.py                     # Abstract RAG system 
├── rag_answer.py             # Example RAG system with HyCE 
├── config.yaml                # System configuration
└── commands.json              # Command definitions
```

### Usage

**Interactive Q&A:**
```bash
python rag_answer.py
```

**Generate Synthetic Test Data:**
```bash
python exp/rag_synthetic_data.py
```

**Run Evaluation on the Synthetic Data:**
```bash
python exp/rag_eval.py
```

### Configuration

The system is configured through `config.yaml`:
```yaml
system:
  max_tokens: 4096
  temperature: 0.7

command_retrieval:
  top_k: 5
  min_score: -100

reranking:
  enabled: true
  top_k: 5
  min_score: -100

execution:
  timeout: 30
  max_output_length: 1000

paths:
  commands_file: commands.json
  artifacts_dir: artifacts
  web_data_file: web_data.json
```

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


