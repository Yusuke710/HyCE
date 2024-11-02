# web_scrape.py

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import json

def normalize_url(url):
    """
    Normalizes the URL by removing the fragment and query parts,
    and ensuring there is no trailing slash.
    """
    parsed = urlparse(url)
    # Remove fragment and query, and standardize the path
    path = parsed.path.rstrip('/')
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))
    return normalized

def save_web_data(url, output_dir='web_data'):
    """
    Fetches the web page from the given URL, extracts the content from linked pages,
    and saves the data to a JSON file.
    Each chunk (paragraph) is treated individually.
    """
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

                    sections = [section.get_text(strip=True) for section in link_soup.find_all('p')]
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

def load_data_chunks(json_file_path):
    """
    Loads the web data from a JSON file.
    """
    if not os.path.exists(json_file_path):
        print(f"The file {json_file_path} does not exist.")
        return []
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data
