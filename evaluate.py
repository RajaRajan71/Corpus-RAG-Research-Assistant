import json
import numpy as np
from rouge_score import rouge_scorer
from summarizer import GeminiSummarizer

def evaluate_pipeline():
    print("==========================================")
    print("    Initializing RAG Evaluation Suite    ")
    print("==========================================\n")
    
    # Initialize summarizer (which internally loads HybridSearchEngine)
    summarizer = GeminiSummarizer()
    
    # 1. Define Test Queries with Expected Ground-Truth Keywords/Topics in Corpus
    test_set = [
        {
            "query": "What is alchemy?",
            "expected_title": "Alchemy"
        },
        {
            "query": "Who was Aristotle?",
            "expected_title": "Aristotle"
        },
        {
            "query": "What is altruism?",
            "expected_title": "Altruism"
        }
    ]
    
    search_hits = 0
    reciprocal_ranks = []
    
    # Initialize ROUGE Scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    print("\n--- Starting Evaluation Loop ---")
    for idx, item in enumerate(test_set):
        query = item["query"]
        expected = item["expected_title"].lower()
        print(f"\n[Test Case {idx + 1}/{len(test_set)}] Query: '{query}'")
        
        # A. Evaluate Retrieval
        summary, sources = summarizer.generate_summary(query, top_n=3, length_option="medium")
        
        found_rank = 0
        for rank, source in enumerate(sources):
            if expected in source["title"].lower():
                found_rank = rank + 1
                break
                
        if found_rank > 0:
            search_hits += 1
            rr = 1.0 / found_rank
            reciprocal_ranks.append(rr)
            print(f"  ✓ Target document found at Rank {found_rank} (Reciprocal Rank: {rr:.2f})")
        else:
            reciprocal_ranks.append(0.0)
            print(f"  ✗ Target document '{item['expected_title']}' not in Top 3 search results")
            
        # B. Evaluate Summarization Quality against Context Grounding
        context_text = " ".join([s["content"] for s in sources])
        scores = rouge.score(context_text, summary)
        
        for key in rouge_scores:
            f1_score = scores[key].fmeasure
            rouge_scores[key].append(f1_score)
            
        print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.4f} | ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

    # Calculate Aggregate Metrics
    hit_rate = (search_hits / len(test_set)) * 100
    mrr = np.mean(reciprocal_ranks)
    avg_rouge1 = np.mean(rouge_scores['rouge1'])
    avg_rouge2 = np.mean(rouge_scores['rouge2'])
    avg_rougeL = np.mean(rouge_scores['rougeL'])
    
    print("\n==========================================")
    print("         FINAL EVALUATION REPORT          ")
    print("==========================================")
    print(f"Total Test Cases      : {len(test_set)}")
    print(f"Search Hit Rate (@3)  : {hit_rate:.2f}%")
    print(f"Mean Reciprocal Rank  : {mrr:.4f}")
    print("------------------------------------------")
    print(f"Average ROUGE-1 F1    : {avg_rouge1:.4f}")
    print(f"Average ROUGE-2 F1    : {avg_rouge2:.4f}")
    print(f"Average ROUGE-L F1    : {avg_rougeL:.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    evaluate_pipeline()