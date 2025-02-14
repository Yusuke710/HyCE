import os
import json
import subprocess
import faiss
import concurrent.futures
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import time

@dataclass
class CommandResult:
    """Represents the result of a command execution"""
    command: str
    description: str
    output: str
    success: bool
    error: Optional[str] = None

class BaseCommandRetriever(ABC):
    """Abstract base class for command retrievers"""
    
    @abstractmethod
    def find_relevant_commands(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant commands for a query"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str) -> CommandResult:
        """Execute a command and return its result"""
        pass
    
    @abstractmethod
    def process_contexts(self, query: str, doc_contexts: List[Dict[str, Any]], 
                        top_k_docs: int = 5, top_k_commands: int = 3, 
                        final_top_k: int = 5, debug: bool = False) -> List[Dict[str, Any]]:
        """Process and combine document and command contexts"""
        pass

class HyCE(BaseCommandRetriever):
    """
    Hypothetical Command Embeddings (HyCE) manager.
    A pluggable component that can be added to any RAG system to enable command execution.
    """
    
    def __init__(self, embedder, reranker, commands_file='commands.json', config=None):
        super().__init__()
        self.embedder = embedder
        self.reranker = reranker
        self.commands_file = commands_file
        self.config = config or {}
        
        # Get config values with defaults
        self.top_k = self.config.get('command_retrieval', {}).get('top_k', 5)
        self.min_score = self.config.get('command_retrieval', {}).get('min_score', 0.5)
        self.rerank_top_k = self.config.get('reranking', {}).get('top_k', 3)
        self.rerank_min_score = self.config.get('reranking', {}).get('min_score', 0.7)
        self.timeout = self.config.get('execution', {}).get('timeout', 30)
        self.max_output_length = self.config.get('execution', {}).get('max_output_length', 1000)
        
        # Initialize commands and index
        self.commands = self._load_commands(self.commands_file)
        self.command_index = None
        self._build_command_index()
    
    def _load_commands(self, file_path: str) -> Dict[str, str]:
        """Load commands and their descriptions from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Commands file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_command_index(self) -> None:
        """Build FAISS index for command descriptions"""
        descriptions = list(self.commands.values())
        embeddings = self.embedder.encode(descriptions)
        
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.cpu().numpy()
        else:  # numpy array
            embeddings_np = embeddings
        
        # Build FAISS index
        dim = embeddings_np.shape[1]
        self.command_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings_np)
        self.command_index.add(embeddings_np)
    
    def execute_command(self, command: str) -> CommandResult:
        """Safely execute a command and return its output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return CommandResult(
                command=command,
                description=self.commands.get(command, ""),
                output=result.stdout.strip(),
                success=result.returncode == 0,
                error=result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                command=command,
                description=self.commands.get(command, ""),
                output="",
                success=False,
                error=f"Command timed out after {self.timeout} seconds"
            )
        except Exception as e:
            return CommandResult(
                command=command,
                description=self.commands.get(command, ""),
                output="",
                success=False,
                error=str(e)
            )
    
    def find_relevant_commands(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant commands for a query"""
        query_embedding = self.embedder.encode([query])
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        faiss.normalize_L2(query_embedding)
        distances, indices = self.command_index.search(query_embedding, top_k)
        
        command_names = list(self.commands.keys())
        results = []
        for idx, score in zip(indices[0], distances[0]):
            cmd_name = command_names[idx]
            results.append({
                'url': 'command',
                'chunk': self.commands[cmd_name],
                'command': cmd_name,
                'score': float(score)
            })
        
        return results
    
    def process_contexts(self, query: str, doc_contexts: List[Dict[str, Any]], 
                        top_k_docs: int = 5, top_k_commands: int = 3, 
                        final_top_k: int = 5, debug: bool = False) -> List[Dict[str, Any]]:
        """Process and combine document and command contexts"""
        
        if debug:
            print("\n=== Command Retrieval ===")
        start_time = time.time()
        
        # Create thread pool for parallel retrieval
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit command retrieval
            cmd_future = executor.submit(self.find_relevant_commands, query, top_k_commands)
            cmd_contexts = cmd_future.result()
        
        if debug:
            cmd_time = time.time() - start_time
            print(f"Command retrieval took: {cmd_time:.2f} seconds")
            print(f"Found {len(cmd_contexts)} commands:")
            for cmd in cmd_contexts:
                print(f"- {cmd['command']} (Score: {cmd['score']:.3f})")
        
        if debug:
            print("\n=== Reranking Process ===")
        start_time = time.time()
        
        # Combine all contexts
        all_contexts = doc_contexts + cmd_contexts
        texts = [chunk['chunk'] for chunk in all_contexts]
        
        # Rerank using cross-encoder
        reranked = self.reranker.rerank(query, texts)
        for chunk, (_, score) in zip(all_contexts, reranked):
            chunk['cross_encoder_score'] = float(score)
        
        # Sort by score
        ranked_contexts = sorted(
            all_contexts, 
            key=lambda x: x['cross_encoder_score'], 
            reverse=True
        )[:final_top_k]
        
        if debug:
            rerank_time = time.time() - start_time
            print(f"Reranking took: {rerank_time:.2f} seconds")
            print("Reranked results:")
            for ctx in ranked_contexts:
                if ctx.get('url') == 'command':
                    print(f"- {ctx['command']} (Score: {ctx['cross_encoder_score']:.3f})")
                else:
                    print(f"- {ctx.get('url', 'unknown')} (Score: {ctx['cross_encoder_score']:.3f})")
        
        # Execute commands that meet threshold
        if debug:
            print("\n=== Command Execution ===")
        start_time = time.time()
        
        min_score = self.config.get('reranking', {}).get('min_score', -100)
        for context in ranked_contexts:
            if (context.get('url') == 'command' and 
                context.get('cross_encoder_score', 0) >= min_score):
                result = self.execute_command(context['command'])
                if result.success:
                    context['command_output'] = result.output
        
        if debug:
            exec_time = time.time() - start_time
            print(f"Command execution took: {exec_time:.2f} seconds")
        
        return ranked_contexts
    
    def get_command_contexts(self) -> List[Dict[str, Any]]:
        """Get all commands as contexts for initial corpus building"""
        return [{
            'url': 'command',
            'chunk': desc,
            'command': cmd
        } for cmd, desc in self.commands.items()]

__all__ = ['CommandResult', 'BaseCommandRetriever', 'HyCE'] 