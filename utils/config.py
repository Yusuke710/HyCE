import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
CONFIG = load_config()

# Get system settings
SYSTEM_CONFIG = CONFIG.get('system', {})
MAX_NUM_TOKENS = SYSTEM_CONFIG.get('max_tokens', 4096)
DEFAULT_TEMPERATURE = SYSTEM_CONFIG.get('temperature', 0.7)

# Get available models
AVAILABLE_MODELS = {
    name: config for name, config in CONFIG.get('models', {}).items()
    if name != 'default'
}

# Get default model with fallback
models_config = CONFIG.get('models', {})
DEFAULT_MODEL = models_config.get('default')
if not DEFAULT_MODEL or DEFAULT_MODEL not in AVAILABLE_MODELS:
    # Fallback to first available model if default is not set or invalid
    DEFAULT_MODEL = next(iter(AVAILABLE_MODELS)) if AVAILABLE_MODELS else None
    if not DEFAULT_MODEL:
        raise ValueError("No models available in configuration")

# Get list of available model names
AVAILABLE_LLMS = list(AVAILABLE_MODELS.keys())

def get_embedding_config(name: str = None) -> Dict[str, Any]:
    """Get embedding model configuration"""
    embeddings = CONFIG.get('embeddings', {})
    if name is None:
        name = embeddings.get('default')
    if not name:
        raise ValueError("No default embedding model specified")
    config = embeddings.get(name)
    if not config:
        raise ValueError(f"Embedding model {name} not found in configuration")
    return config

def get_reranker_config(name: str = None) -> Dict[str, Any]:
    """Get reranker model configuration"""
    rerankers = CONFIG.get('rerankers', {})
    if name is None:
        name = rerankers.get('default')
    if not name:
        raise ValueError("No default reranker model specified")
    config = rerankers.get(name)
    if not config:
        raise ValueError(f"Reranker model {name} not found in configuration")
    return config

def get_system_config() -> Dict[str, Any]:
    """Get system settings"""
    return CONFIG.get('system', {})

def get_paths_config() -> Dict[str, Any]:
    """Get file paths"""
    return CONFIG.get('paths', {}) 