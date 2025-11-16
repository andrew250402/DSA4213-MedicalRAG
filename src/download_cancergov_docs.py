"""
Script to download cancer.gov documents from XML files.
Saves raw HTML and extracted plain text versions.
"""

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
import logging
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_xml_files(xml_dir: str) -> List[Tuple[str, str]]:
    """
    Parse all XML files in directory to extract document IDs and URLs.

    Args:
        xml_dir: Path to directory containing XML files

    Returns:
        List of tuples containing (document_id, url)
    """
    documents = []
    xml_path = Path(xml_dir)

    if not xml_path.exists():
        logger.error(f"Directory {xml_dir} does not exist")
        return documents

    xml_files = sorted(xml_path.glob("*.xml"))
    logger.info(f"Found {len(xml_files)} XML files")

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            doc_id = root.get('id')
            url = root.get('url')

            if doc_id and url:
                documents.append((doc_id, url))
            else:
                logger.warning(f"Missing id or url in {xml_file.name}")

        except Exception as e:
            logger.error(f"Error parsing {xml_file.name}: {e}")

    logger.info(f"Successfully parsed {len(documents)} documents")
    return documents


def download_html(url: str) -> str:
    """
    Download HTML content from URL.

    Args:
        url: URL to download from

    Returns:
        HTML content as string
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download from {url}: {str(e)}")


def extract_main_content(html: str) -> str:
    """
    Extract main article content from HTML.
    
    Args:
        html: Raw HTML string
        
    Returns:
        Plain text content
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the article element
    article = soup.find('article')
    
    if not article:
        # Fallback if no article tag found
        raise Exception("No <article> element found in HTML")
    
    # Remove the "About This" section if it exists
    about_section = article.find('section', id='_AboutThis_1')
    if about_section:
        about_section.decompose()
    
    # Remove footer elements within the article
    for footer in article.find_all('footer'):
        footer.decompose()
    
    # Get text and clean up whitespace
    text = article.get_text(separator='\n', strip=True)
    
    # Remove excessive newlines and clean up spacing
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Skip empty lines
        if line:
            # Remove multiple spaces
            line = ' '.join(line.split())
            lines.append(line)
    
    return ' '.join(lines)


def process_documents(
    documents: List[Tuple[str, str]],
    raw_dir: str,
    plain_dir: str,
    overwrite: bool = False
):
    """
    Download and process all documents.

    Args:
        documents: List of (doc_id, url) tuples
        raw_dir: Directory to save raw HTML files
        plain_dir: Directory to save plain text files
        overwrite: Whether to overwrite existing files
    """
    # Create directories
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(plain_dir).mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    error_count = 0
    failed_docs = []

    for doc_id, url in documents:
        raw_path = Path(raw_dir) / f"{doc_id}.html"
        plain_path = Path(plain_dir) / f"{doc_id}.txt"

        # Skip if files exist and overwrite is False
        if not overwrite and raw_path.exists() and plain_path.exists():
            logger.info(f"Skipping {doc_id} (already exists)")
            skip_count += 1
            continue

        try:
            logger.info(f"Downloading {doc_id} from {url}")

            # Download HTML
            html = download_html(url)

            # Save raw HTML
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.debug(f"Saved raw HTML to {raw_path}")

            # Extract and save plain text
            plain_text = extract_main_content(html)
            with open(plain_path, 'w', encoding='utf-8') as f:
                f.write(plain_text)
            logger.debug(f"Saved plain text to {plain_path}")

            success_count += 1
            logger.info(f"✓ Successfully processed {doc_id}")

        except Exception as e:
            logger.error(f"✗ Error processing {doc_id}: {e}")
            failed_docs.append((doc_id, url, str(e)))
            error_count += 1

    logger.info(f"\nSummary:")
    logger.info(f"  Successfully processed: {success_count}")
    logger.info(f"  Skipped (already exist): {skip_count}")
    logger.info(f"  Errors: {error_count}")

    if failed_docs:
        logger.info(f"\nFailed documents:")
        for doc_id, url, error in failed_docs:
            logger.info(f"  - {doc_id}: {url}")
            logger.info(f"    Error: {error}")


def main():
    parser = argparse.ArgumentParser(
        description='Download cancer.gov documents from XML files'
    )
    parser.add_argument(
        '--xml-dir',
        default='1_CancerGov_QA',
        help='Directory containing XML files (default: 1_CancerGov_QA)'
    )
    parser.add_argument(
        '--output-dir',
        default='data_v2',
        help='Output directory (default: data_v2)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files (default: False)'
    )

    args = parser.parse_args()

    # Parse XML files
    documents = parse_xml_files(args.xml_dir)

    if not documents:
        logger.error("No documents found to process")
        return

    # Set up output directories
    raw_dir = os.path.join(args.output_dir, 'raw')
    plain_dir = os.path.join(args.output_dir, 'plain')

    # Process documents
    process_documents(documents, raw_dir, plain_dir, args.overwrite)


if __name__ == '__main__':
    main()
