# Empty file to make utils a package 
from .web_scraper import (
    load_data_chunks,
    save_web_data_by_tag,
    save_web_data_by_llm,
    normalize_url
)
from .llm import get_response_from_llm, extract_json_between_markers

__all__ = [
    'load_data_chunks',
    'save_web_data_by_tag',
    'save_web_data_by_llm',
    'normalize_url',
    'get_response_from_llm',
    'extract_json_between_markers'
] 