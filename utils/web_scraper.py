import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import json
import re
import openai
from utils.llm import get_response_from_llm, extract_json_between_markers

def normalize_url(url):
    """Normalizes the URL by removing fragments and queries"""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    return urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))

def save_web_data_by_tag(url, output_dir='artifacts'):
    """Scrapes web data using HTML tags"""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch the main URL {url}: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    anchors = soup.find_all('a')

    data = []
    os.makedirs(output_dir, exist_ok=True)
    visited_urls = set()

    for anchor in anchors:
        href = anchor.get('href')
        if href and not href.startswith('#'):
            full_url = urljoin(url, href)
            normalized_url = normalize_url(full_url)
            if normalized_url not in visited_urls:
                visited_urls.add(normalized_url)
                try:
                    link_response = requests.get(full_url)
                    link_response.raise_for_status()
                    link_soup = BeautifulSoup(link_response.content, 'html.parser')

                    sections = [section.get_text(strip=True) 
                              for section in link_soup.find_all('p')]
                    if sections:
                        for idx, chunk in enumerate(sections):
                            data.append({
                                'url': normalized_url,
                                'chunk_idx': idx,
                                'chunk': chunk
                            })
                        print(f'Saved content for {normalized_url}')
                    else:
                        print(f'No content found at {normalized_url}')
                except requests.RequestException as e:
                    print(f'Failed to fetch {normalized_url}: {e}')

    json_file_path = os.path.join(output_dir, 'web_data.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f'All data saved to {json_file_path}')

def save_web_data_by_llm(url, output_dir='artifacts'):
    """Scrapes web data using LLM for semantic chunking"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch the main URL {url}: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    anchors = soup.find_all('a')

    data = []
    visited_urls = set()

    for anchor in anchors:
        href = anchor.get('href')
        if href and not href.startswith('#'):
            full_url = urljoin(url, href)
            normalized_url = normalize_url(full_url)
            if normalized_url not in visited_urls:
                visited_urls.add(normalized_url)
                try:
                    link_response = requests.get(full_url)
                    link_response.raise_for_status()
                    link_soup = BeautifulSoup(link_response.content, 'html.parser')

                    paragraphs = [p.get_text() for p in link_soup.find_all("p")]
                    text_content = "\n".join(paragraphs)
                    full_text = text_content.strip()

                    # Use LLM to chunk text semantically
                    chunks = chunk_text_semantically(full_text)
                    if chunks:
                        for idx, chunk in enumerate(chunks):
                            data.append({
                                'url': normalized_url,
                                'chunk_idx': idx,
                                'chunk': chunk
                            })
                        print(f'Saved content for {normalized_url}')
                    else:
                        print(f'No content found at {normalized_url}')
                except requests.RequestException as e:
                    print(f'Failed to fetch {normalized_url}: {e}')

    json_file_path = os.path.join(output_dir, 'web_data.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f'All data saved to {json_file_path}')

def chunk_text_semantically(text):
    """Uses LLM to split text into semantic chunks"""
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