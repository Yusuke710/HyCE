import os
import json
import subprocess
import faiss
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class CommandResult:
    """Represents the result of a command execution"""
    command: str
    description: str
    output: str
    success: bool
    error: Optional[str] = None

class HyCE:
    """Hypothetical Command Embeddings (HyCE) manager"""
    
    def __init__(self, embedding_model, cross_encoder, commands_file: str = 'commands.json', 
                 timeout: int = 10, similarity_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder
        self.timeout = timeout
        self.similarity_threshold = similarity_threshold
        self.commands = self._load_commands(commands_file)
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
        embeddings = self.embedding_model.encode(
            descriptions,
            batch_size=4,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Convert to numpy and normalize
        embeddings_np = embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        
        # Build FAISS index
        dim = embeddings_np.shape[1]
        self.command_index = faiss.IndexFlatIP(dim)
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
        """Find most relevant commands for a query with their descriptions"""
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)
        
        # Search index
        distances, indices = self.command_index.search(query_embedding_np, top_k)
        
        # Get command info with scores
        command_names = list(self.commands.keys())
        results = []
        for idx, score in zip(indices[0], distances[0]):
            cmd_name = command_names[idx]
            results.append({
                'url': 'command',
                'chunk': self.commands[cmd_name],
                'command': cmd_name,
                'bi_encoder_score': float(score)
            })
        
        return results

    def rerank_combined_results(self, query: str, doc_chunks: List[Dict[str, Any]], 
                              cmd_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank combined document and command chunks using cross-encoder.
        """
        # Combine all chunks
        all_chunks = doc_chunks + cmd_chunks
        
        # Prepare cross-encoder inputs
        cross_inputs = []
        for chunk in all_chunks:
            cross_inputs.append((query, chunk['chunk']))
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(cross_inputs)
        
        # Add scores to chunks
        for chunk, score in zip(all_chunks, cross_scores):
            chunk['cross_encoder_score'] = float(score)
        
        # Sort by cross-encoder score
        ranked_chunks = sorted(all_chunks, 
                             key=lambda x: x['cross_encoder_score'], 
                             reverse=True)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in ranked_chunks:
            identifier = (chunk.get('url'), chunk['chunk'])
            if identifier not in seen:
                seen.add(identifier)
                unique_chunks.append(chunk)
                if len(unique_chunks) >= top_k:
                    break
        
        return unique_chunks

    def process_contexts(self, query: str, doc_contexts: List[Dict[str, Any]], 
                        top_k_docs: int = 5, top_k_commands: int = 3, 
                        final_top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank both documents and commands, then execute relevant commands.
        """
        # Get relevant commands separately
        cmd_contexts = self.find_relevant_commands(query, top_k=top_k_commands)
        
        # Rerank combined results
        ranked_contexts = self.rerank_combined_results(
            query=query,
            doc_chunks=doc_contexts,
            cmd_chunks=cmd_contexts,
            top_k=final_top_k
        )
        
        # Execute commands that made it into top results and meet threshold
        for context in ranked_contexts:
            if (context.get('url') == 'command' and 
                context.get('cross_encoder_score', 0) >= self.similarity_threshold):
                
                result = self.execute_command(context['command'])
                if result.success:
                    context['command_output'] = result.output
        
        return ranked_contexts
    
    def get_command_contexts(self) -> List[Dict[str, Any]]:
        """Get all commands as contexts for initial corpus building"""
        contexts = []
        for cmd, desc in self.commands.items():
            contexts.append({
                'url': 'command',
                'chunk': desc,
                'command': cmd
            })
        return contexts 