import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import json
import re
import openai
from utils.llm import get_response_from_llm, extract_json_between_markers
import time

def normalize_url(url):
    """Normalizes the URL by removing fragments and queries"""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    return urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))

def scrape_documentation(output_path: str) -> None:
    """
    Scrape Katana documentation using LLM-based semantic chunking.
    
    Args:
        output_path: Path where the scraped data will be saved
    """
    print("Starting documentation scraping with LLM-based chunking...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Start with the main Katana documentation URL
    base_url = "https://docs.restech.unsw.edu.au/"
    
    try:
        # Use the LLM-based scraping
        save_web_data_by_llm(base_url, os.path.dirname(output_path))
        
        # Rename the output file if needed
        temp_path = os.path.join(os.path.dirname(output_path), 'web_data.json')
        if temp_path != output_path and os.path.exists(temp_path):
            os.rename(temp_path, output_path)
            
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        raise

def save_web_data_by_llm(url, output_dir='artifacts'):
    """Scrapes web data using LLM for semantic chunking"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch the main URL {url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    data = []
    visited_urls = set()

    # Process main page first
    main_content = extract_page_content(soup)
    if main_content:
        chunks = chunk_text_semantically(main_content)
        if chunks:
            for idx, chunk in enumerate(chunks):
                data.append({
                    'url': url,
                    'chunk_idx': idx,
                    'chunk': chunk,
                    'title': soup.title.string if soup.title else url
                })
            print(f'✓ Saved content for main page')

    # Process all links
    for link in links:
        href = link.get('href')
        if not href or href.startswith('#'):
            continue

        full_url = urljoin(url, href)
        normalized_url = normalize_url(full_url)

        # Only process Katana documentation URLs
        if not normalized_url.startswith('https://docs.restech.unsw.edu.au/'):
            continue

        if normalized_url in visited_urls:
            continue

        visited_urls.add(normalized_url)
        
        try:
            print(f"Scraping: {normalized_url}")
            page_response = requests.get(normalized_url, timeout=30)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            content = extract_page_content(page_soup)
            if content:
                chunks = chunk_text_semantically(content)
                if chunks:
                    for idx, chunk in enumerate(chunks):
                        data.append({
                            'url': normalized_url,
                            'chunk_idx': idx,
                            'chunk': chunk,
                            'title': page_soup.title.string if page_soup.title else normalized_url
                        })
                    print(f'✓ Content saved and chunked')
                else:
                    print(f'No chunks generated for {normalized_url}')
            
            # Be nice to the server
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing {normalized_url}: {str(e)}")
            continue

    # Save all data
    json_file_path = os.path.join(output_dir, 'web_data.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nScraped and chunked {len(data)} segments from {len(visited_urls)} pages")
    print(f"Data saved to: {json_file_path}")

def extract_page_content(soup):
    """Extract main content from a page"""
    content = ""
    
    # Try to find main content container
    main_content = soup.find(['main', 'article', 'div', 'section'], 
                           class_=['content', 'main-content', 'article-content'])
    
    if main_content:
        # Extract from main content area
        for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'pre', 'code', 'ul', 'ol']):
            text = element.get_text().strip()
            if text:
                if element.name in ['h1', 'h2', 'h3', 'h4']:
                    content += f"\n\n{text}\n"
                elif element.name in ['pre', 'code']:
                    content += f"\n```\n{text}\n```\n"
                else:
                    content += f"{text}\n"
    else:
        # Fallback to basic content extraction
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'pre', 'code']):
            text = element.get_text().strip()
            if text:
                content += text + "\n\n"
    
    return content.strip()

def chunk_text_semantically(text):
    """Uses LLM to split text into semantic chunks"""
    if not text or len(text) < 100:  # Skip very short texts
        return []
        
    prompt = (
        "Please split the following text into semantically coherent chunks. "
        "Each chunk should represent a coherent idea or topic. "
        "Omit short parts that are not relevant to the topic. "
        "Return the chunks as a JSON array of strings.\n\nText:\n" + text
    )
    
    system_message = "You are a helpful assistant that splits text into semantically coherent chunks."
    
    try:
        content, _ = get_response_from_llm(
            msg=prompt,
            client=openai,
            model='gpt-4o-2024-08-06',
            system_message=system_message,
            print_debug=False
        )
        chunks = extract_json_between_markers(content)
        if chunks is None:
            print("Failed to extract JSON from LLM output.")
            return []
        return chunks
    except Exception as e:
        print(f"Error during semantic chunking: {e}")
        return []

def load_data_chunks(json_file_path):
    """Loads the web data from a JSON file"""
    if not os.path.exists(json_file_path):
        print(f"The file {json_file_path} does not exist.")
        return []
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

if __name__ == "__main__":
    scrape_documentation("artifacts/web_data.json") 