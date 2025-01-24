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

## Contributing
Feel free to submit issues and enhancement requests.


