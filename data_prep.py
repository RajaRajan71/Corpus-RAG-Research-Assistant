import os
import json
import re
from datasets import load_dataset
from tqdm import tqdm

def clean_text(text: str) -> str:
    """Cleans raw Wikipedia text by removing references, extra whitespaces, and unwanted artifacts."""
    # Remove citation brackets like [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks of a specified character length."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks

def prepare_wikipedia_corpus(num_articles: int = 50, output_file: str = "wiki_corpus.json"):
    print(f"Loading {num_articles} Wikipedia articles from Hugging Face...")
    
    # FIX: Using the modern wikimedia/wikipedia dataset path instead of the legacy script loader
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    
    corpus = []
    doc_id = 0
    
    for article in tqdm(dataset['train'].take(num_articles), total=num_articles):
        title = article['title']
        raw_text = article['text']
        
        # Clean text
        cleaned = clean_text(raw_text)
        if len(cleaned) < 200:  # Skip articles that are too short
            continue
            
        # Chunk text
        text_chunks = chunk_text(cleaned, chunk_size=500, overlap=50)
        
        for chunk_id, chunk_content in enumerate(text_chunks):
            corpus.append({
                "doc_id": f"doc_{doc_id:04d}",
                "title": title,
                "chunk_id": chunk_id,
                "content": chunk_content,
                "source": "Wikipedia"
            })
        doc_id += 1

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully processed {doc_id} documents into {len(corpus)} total chunks. Saved to {output_file}.")

if __name__ == "__main__":
    prepare_wikipedia_corpus(num_articles=50, output_file="wiki_corpus.json")