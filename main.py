from summarizer import GeminiSummarizer

if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is set in your .env file
    summarizer = GeminiSummarizer()
    query = "Tell me about Aristotle"
    
    summary, sources = summarizer.generate_summary(query, top_n=3, length_option="medium")
    
    print(f"\n=== Query: {query} ===")
    print(f"\n--- Gemini Summary ---\n{summary}")
    print("\n--- Sources Used ---")
    for s in sources:
        print(f"- {s['title']} (Doc ID: {s.get('doc_id', 'N/A')})")