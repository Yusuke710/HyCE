# HyCE

## Overview
This project implements and evaluates a Retrieval-Augmented Generation (RAG) system specifically designed for High Performance Computing (HPC) queries. It combines web-scraped documentation with command-line utilities to provide accurate responses to HPC-related questions.

## Key Features
- Hybrid Context Enhancement (HyCE) approach combining web documentation and command outputs
- Automatic RAG evaluation

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Pipeline Steps

1. Web Scraping and Data Preparation:
```bash
python rag/web_scrape_by_llm.py
```

2. Generate Command Embeddings:
```bash
python command_embedding_hyce.py
```

3. Run RAG System:
```bash
python rag_answer.py
```

### Evaluation

1. Generate Synthetic Test Data:
```bash
python rag_synthetic_data.py
```

2. Run Evaluation:
```bash
python rag_eval.py
```

## File Structure
```
.
├── rag/
│   ├── web_scrape_by_llm.py    # Web scraping utilities
│   ├── text_embedding.py        # Embedding functions
│   └── llm.py                   # LLM interaction utilities
├── command_embedding_hyce.py    # Command embedding generation
├── rag_answer.py               # Main RAG system
├── rag_synthetic_data.py       # Test data generation
└── rag_eval.py                 # Evaluation framework
```

### FAISS compatibility issues with MPS

When runing ```python rag_answer.py```, you may encounter the following error:
```bash
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```
This is due to FAISS not being fully compatible with MPS. To temporarily fix this,
set the following environment parameters:

Disable OpenMP threading via the environment variable: `export OMP_NUM_THREADS=1`
Disable PyTorch MPS device via: `export PYTORCH_MPS_DISABLE=1`
Disable llama.cpp metal via: `export LLAMA_NO_METAL=1`
Source: [https://neuml.github.io/txtai/faq/](https://neuml.github.io/txtai/faq/)

## Contributing
Feel free to submit issues and enhancement requests.


