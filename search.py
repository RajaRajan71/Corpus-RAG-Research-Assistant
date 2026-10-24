import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

class HybridSearchEngine:
    def __init__(self, corpus_path="wiki_corpus.json"):
        print("Loading corpus for search engine...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
            
        self.corpus_texts = [item["content"] for item in self.corpus]
        
        # 1. Initialize BM25
        print("Initializing BM25 keyword index...")
        tokenized_corpus = [doc.lower().split() for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Initialize Vector Store (Using local HuggingFace embeddings to keep it free and fast)
        print("Initializing Vector Embeddings (Sentence Transformers)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Using ChromaDB for local vector persistence
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="wiki_rag_collection")
        
        # Populate ChromaDB if empty
        if self.collection.count() == 0:
            print("Embedding chunks into ChromaDB (this may take a moment)...")
            embeddings = self.embedding_model.encode(self.corpus_texts, show_progress_bar=True).tolist()
            ids = [item["doc_id"] + "_" + str(item["chunk_id"]) for item in self.corpus]
            metadatas = [{"title": item["title"], "chunk_id": item["chunk_id"], "source": item["source"]} for item in self.corpus]
            
            self.collection.add(
                documents=self.corpus_texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print("Vector database populated successfully!")

    def bm25_search(self, query: str, top_k: int = 5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"index": int(idx), "score": float(scores[idx]), "method": "BM25"} for idx in top_indices]

    def vector_search(self, query: str, top_k: int = 5):
        query_vector = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_vector, n_results=top_k)
        
        # Map back to corpus indices
        retrieved = []
        for i in range(len(results['ids'][0])):
            doc_id_chunk = results['ids'][0][i]
            # Find matching index in self.corpus
            matched_idx = next(idx for idx, item in enumerate(self.corpus) if f"{item['doc_id']}_{item['chunk_id']}" == doc_id_chunk)
            retrieved.append({
                "index": matched_idx,
                "score": float(1.0 - results['distances'][0][i]), # converting distance to a relative score
                "method": "Vector"
            })
        return retrieved

    def hybrid_search(self, query: str, top_n: int = 5, k_rrf: int = 60):
        """Combines BM25 and Vector Search using Reciprocal Rank Fusion (RRF)."""
        bm25_results = self.bm25_search(query, top_k=top_n * 2)
        vector_results = self.vector_search(query, top_k=top_n * 2)
        
        rrf_scores = {}
        
        # RRF formula: sum(1 / (k + rank))
        for rank, res in enumerate(bm25_results):
            idx = res["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        for rank, res in enumerate(vector_results):
            idx = res["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        # Sort by highest RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]
        
        final_results = []
        for idx in sorted_indices:
            item = self.corpus[idx]
            final_results.append({
                "doc_id": item["doc_id"],
                "title": item["title"],
                "chunk_id": item["chunk_id"],
                "content": item["content"],
                "rrf_score": rrf_scores[idx]
            })
            
        return final_results

if __name__ == "__main__":
    engine = HybridSearchEngine()
    test_query = "history of computer science"
    print(f"\n--- Testing Search for Query: '{test_query}' ---")
    results = engine.hybrid_search(test_query, top_n=3)
    for r in results:
        print(f"\n[Title: {r['title']} | Doc ID: {r['doc_id']} | Score: {r['rrf_score']:.4f}]")
        print(f"Content: {r['content'][:250]}...")

    # pip install rank-bm25 sentence-transformers chromadb numpy after the code execution

    #python search.py