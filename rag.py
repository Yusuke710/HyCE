import os
import json
import subprocess
import faiss
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils import get_response_from_llm, extract_json_between_markers
from utils.config import get_embedding_config
from hyce import BaseCommandRetriever, HyCE
import time

# Constants
MAX_NUM_TOKENS = 4096

# Embedder Components
class BaseEmbedder(ABC):
    """Abstract base class for text embedders"""
    
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """Encode texts into embeddings"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of the embeddings"""
        pass

class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformer implementation of embedder"""
    
    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
        self.model_name = model_name  # Store the model name
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str], **kwargs) -> torch.Tensor:
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            **kwargs
        )
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API implementation of embedder"""
    
    def __init__(self, client, model="text-embedding-ada-002"):
        self.client = client
        self.model = model
        self._dimension = 1536  # ada-002 dimension
        
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if not isinstance(texts, list):
            texts = [texts]
            
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [r.embedding for r in response.data]
        # Ensure correct shape and dtype for FAISS
        embeddings_array = np.array(embeddings, dtype=np.float32)
        if len(texts) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        return embeddings_array
    
    def get_dimension(self) -> int:
        return self._dimension

# Reranker Components
class BaseReranker(ABC):
    """Abstract base class for rerankers"""
    
    @abstractmethod
    def rerank(self, query: str, texts: List[str], **kwargs) -> List[Tuple[str, float]]:
        """Rerank texts based on query"""
        pass

class CrossEncoderReranker(BaseReranker):
    """Cross-encoder implementation of reranker"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, texts: List[str], **kwargs) -> List[Tuple[str, float]]:
        """Rerank texts using cross-encoder model"""
        pairs = [(query, text) for text in texts]
        
        scores = self.model.predict(pairs)
        
        return list(zip(texts, scores))

# Retriever Components
class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        pass

class FaissRetriever(BaseRetriever):
    """FAISS-based retriever implementation"""
    
    def __init__(self, embedder: BaseEmbedder, corpus: List[Dict[str, Any]], 
                 cache_dir: str = "artifacts/embeddings"):
        self.embedder = embedder
        self.corpus = corpus
        self.cache_dir = cache_dir
        # Create all parent directories
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Created cache directory: {self.cache_dir}")  # Debug print
        self.index = self._build_index()
    
    def _get_cache_path(self, model_name: str) -> str:
        """Get path for cached embeddings"""
        if isinstance(model_name, str):
            safe_name = model_name.replace('/', '_')
        else:
            # For SentenceTransformer, get the model name
            safe_name = model_name.get_sentence_embedding_dimension()
        return os.path.join(self.cache_dir, f"{safe_name}_embeddings.npy")
    
    def _build_index(self) -> faiss.Index:
        """Build FAISS index from corpus, using cache if available"""
        corpus_texts = [doc['chunk'] for doc in self.corpus]
        
        # Get appropriate model name for cache
        if isinstance(self.embedder, OpenAIEmbedder):
            model_name = self.embedder.model
        elif isinstance(self.embedder, SentenceTransformerEmbedder):
            model_name = self.embedder.model_name  # Use our stored model name
        else:
            model_name = 'default'
        
        cache_path = self._get_cache_path(model_name)
        
        print(f"Checking cache at: {cache_path}")  # Debug print
        print(f"Cache directory exists: {os.path.exists(self.cache_dir)}")  # Debug print
        
        # Try to load cached embeddings
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            embeddings_np = np.load(cache_path)
        else:
            print(f"Computing embeddings for {len(corpus_texts)} texts...")
            embeddings = self.embedder.encode(corpus_texts)
            
            # Convert to numpy if needed
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.cpu().numpy()
            else:  # numpy array
                embeddings_np = embeddings
            
            # Save embeddings to cache
            print(f"Saving embeddings to {cache_path}")
            np.save(cache_path, embeddings_np)
        
        # Build FAISS index
        dimension = self.embedder.get_dimension()
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        faiss.normalize_L2(embeddings_np)  # Normalize embeddings
        index.add(embeddings_np)
        
        return index
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for query"""
        # Encode query
        query_embedding = self.embedder.encode([query])
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            doc = self.corpus[int(idx)].copy()
            doc['score'] = float(score)
            results.append(doc)
        
        return results

# RAG System
class StandardRAG:
    """
    Standard RAG implementation that supports pluggable components:
    - Embedder for encoding texts
    - Reranker for improving retrieval quality
    - Retriever for finding relevant documents
    - Command Retriever (like HyCE) for executing commands (optional)
    """
    
    def __init__(self,
                 embedder: BaseEmbedder,
                 reranker: BaseReranker,
                 retriever: BaseRetriever,
                 llm_client: Any,
                 llm_model: str,
                 command_retriever: Optional[BaseCommandRetriever] = None,
                 max_tokens: int = MAX_NUM_TOKENS):
        self.embedder = embedder
        self.reranker = reranker
        self.retriever = retriever
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.command_retriever = command_retriever
        self.max_tokens = max_tokens
    
    def retrieve(self, query: str, top_k: int = 5, debug: bool = False) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts, optionally including HyCE results"""
        total_start = time.time()
        
        if debug:
            print("\n=== Document Retrieval ===")
        start_time = time.time()
        
        # Get initial documents
        doc_contexts = self.retriever.retrieve(query, top_k=top_k)
        
        if debug:
            doc_time = time.time() - start_time
            print(f"Document retrieval took: {doc_time:.2f} seconds")
            print(f"Found {len(doc_contexts)} documents:")
            for doc in doc_contexts:
                print(f"- {doc.get('url', 'unknown')} (Score: {doc.get('score', 'N/A'):.3f})")
        
        # If HyCE is enabled, process with it
        if self.command_retriever:
            contexts = self.command_retriever.process_contexts(
                query=query,
                doc_contexts=doc_contexts,
                top_k_docs=top_k,
                top_k_commands=top_k,
                final_top_k=top_k,
                debug=debug
            )
        else:
            contexts = doc_contexts
        
        if debug:
            total_time = time.time() - total_start
            print(f"\nTotal retrieval process took: {total_time:.2f} seconds")
        
        return contexts
    
    def format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """Format contexts into a string for the LLM"""
        context = ""
        for i, chunk_info in enumerate(contexts):
            if chunk_info.get('url') == 'command':
                context += f"Command {i+1} Explanation:\n{chunk_info['chunk']}\n"
                if 'command_output' in chunk_info:
                    context += f"Command Output:\n{chunk_info['command_output']}\n"
                context += "\n"
            else:
                context += f"Document {i+1}:\n{chunk_info['chunk']}\n\n"
        
        # Limit context length
        max_chars = 4 * self.max_tokens
        if len(context) > max_chars:
            context = context[:max_chars]
        
        return context
    
    def query(self, query: str, top_k: int = 5, cot: bool = False, 
             debug: bool = False) -> str:
        """Full RAG pipeline: retrieve + generate"""
        # Retrieval phase
        contexts = self.retrieve(query, top_k=top_k, debug=debug)
        
        # LLM phase
        if debug:
            print("\n=== LLM Processing ===")
        start_time = time.time()
        
        context = self.format_context(contexts)
        system_message = "You are an assistant that provides accurate and concise answers based on the provided documents."
        
        if cot:
            user_message = (
                f"Answer the following question based on the provided documents.\n"
                f"Question: {query}\n\n"
                f"Documents:\n{context}\n\n"
                f"Provide your thinking for each step, and then provide the answer in the following **JSON format**:\n"
                "```json\n"
                "{\n"
                "    \"reasons\": <reasons>,\n"
                "    \"answer\": <answer>\n"
                "}\n"
                "```"
            )
        else:
            user_message = f"Answer the following question based on the provided documents.\n\nQuestion: {query}\n\nDocuments:\n{context}"

        try:
            result = get_response_from_llm(
                msg=user_message,
                client=self.llm_client,
                model=self.llm_model,
                system_message=system_message,
                print_debug=debug
            )
            
            if debug:
                llm_time = time.time() - start_time
                print(f"LLM processing took: {llm_time:.2f} seconds")
            
            if result is None:
                return "Error: Failed to get response from LLM"
            
            content, _ = result
            
            if cot:
                response_json = extract_json_between_markers(content)
                if response_json is None:
                    return "Failed to parse LLM response"
                return response_json.get("answer", "No answer found in response")
            else:
                return content or "No response generated"
        except Exception as e:
            return f"Error generating response: {str(e)}"

def create_standard_rag(
    corpus: List[Dict[str, Any]], 
    llm_client: Any,
    llm_model: str,
    embedding_type: str = "sentence-transformer",
    use_hyce: bool = True,
    commands_file: str = 'commands.json',
    hyce_config: Dict[str, Any] = None
) -> StandardRAG:
    """Factory function to create a standard RAG system"""
    
    # Get embedding config
    embedding_config = get_embedding_config(embedding_type)
    
    # Create embedder based on type
    if embedding_type == "openai-ada":
        embedder = OpenAIEmbedder(
            client=llm_client,
            model=embedding_config.get('model', 'text-embedding-ada-002')
        )
    else:  # default to sentence-transformer
        embedder = SentenceTransformerEmbedder(
            model_name=embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
    
    reranker = CrossEncoderReranker()
    retriever = FaissRetriever(embedder=embedder, corpus=corpus)
    
    command_retriever = None
    if use_hyce:
        command_retriever = HyCE(
            embedder=embedder,
            reranker=reranker,
            commands_file=commands_file,
            config=hyce_config or {}
        )
    
    return StandardRAG(
        embedder=embedder,
        reranker=reranker,
        retriever=retriever,
        llm_client=llm_client,
        llm_model=llm_model,
        command_retriever=command_retriever
    )

# Web Scraping Utilities
def load_data_chunks(json_file_path):
    """Loads the web data from a JSON file"""
    if not os.path.exists(json_file_path):
        print(f"The file {json_file_path} does not exist.")
        return []
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data 