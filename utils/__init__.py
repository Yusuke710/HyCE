# Empty file to make utils a package 
from .web_scraper import (
    scrape_documentation,
    save_web_data_by_llm,
    load_data_chunks
)
from .llm import get_response_from_llm, extract_json_between_markers

__all__ = [
    'scrape_documentation',
    'save_web_data_by_llm',
    'load_data_chunks',
    'get_response_from_llm',
    'extract_json_between_markers'
] 