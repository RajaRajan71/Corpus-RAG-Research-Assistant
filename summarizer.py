import os
from dotenv import load_dotenv
from google import genai
from search import HybridSearchEngine

load_dotenv()

class GeminiSummarizer:
    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API Key is missing. Please set GEMINI_API_KEY in your .env file.")
            
        print("Initializing Gemini API Client...")
        self.client = genai.Client(api_key=key)
        
        # Initialize hybrid search engine
        self.search_engine = HybridSearchEngine()

    def generate_summary(self, query: str, top_n: int = 3, length_option: str = "medium") -> tuple[str, list]:
        # 1. Retrieve relevant chunks using hybrid search
        retrieved_docs = self.search_engine.hybrid_search(query, top_n=top_n)
        
        # 2. Format context from retrieved documents
        context_blocks = []
        for i, doc in enumerate(retrieved_docs):
            context_blocks.append(f"Source [{i+1}] (Title: {doc['title']}):\n{doc['content']}")
        
        combined_context = "\n\n".join(context_blocks)
        
        # 3. Define length instructions
        length_instructions = {
            "short": "Provide a brief, high-level summary in 2-3 concise sentences.",
            "medium": "Provide a coherent summary consisting of 1-2 well-structured paragraphs capturing the core essence.",
            "detailed": "Provide a comprehensive, detailed summary broken down into key bullet points and sections explaining the findings."
        }
        instruction = length_instructions.get(length_option, length_instructions["medium"])

        # 4. Construct prompt for Gemini
        prompt = f"""You are an expert research assistant. Answer the user's query accurately using ONLY the provided context excerpts below. Do not make up facts outside the context.

User Query: "{query}"

{instruction}

Context Excerpts:
{combined_context}

Summary:"""
        
        # 5. Execute API request using Gemini 3.5 Flash
        print(f"\nGenerating summary via Gemini 3.5 Flash API (Length: {length_option})...")
        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        
        summary = response.text.strip()
        return summary, retrieved_docs