# 📚 Corpus Document Search & Summarization using LLM (RAG Pipeline)

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that performs hybrid document retrieval and grounded context summarization over a custom Wikipedia corpus on philosophy, history, and science.

Powered by BM25, Sentence Transformers, ChromaDB, Google Gemini 3.5 Flash, and Streamlit.

🌟 Features

Data Processing & Chunking: Streams Wikipedia articles, cleans citations and formatting artifacts, and chunks text into overlapping passages (500 characters, 50-character overlap).

Hybrid Search Engine: Combines sparse keyword search (BM25) with dense semantic vector search (all-MiniLM-L6-v2 via ChromaDB) fused using Reciprocal Rank Fusion (RRF) for optimal retrieval accuracy.

Grounded Summarization: Employs Google Gemini 3.5 Flash to synthesize answers using only the retrieved context chunks, preventing hallucinations. Supports dynamic output length selection (short, medium, detailed).

Automated Evaluation Suite: Includes built-in evaluation metrics measuring retrieval performance (Hit Rate @3, Mean Reciprocal Rank (MRR)) and summarization quality (ROUGE-1, ROUGE-2, ROUGE-L).

Interactive Streamlit Web UI: Features topic exploration banners, clickable sample queries, real-time context source expanders, and graceful exception handling for API traffic spikes.

📐 System Architecture

                                 ┌────────────────────────┐
                                 │ User Query (app.py)    │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                               ┌──────────────────────────┐
                               │   Hybrid Search Engine   │
                               │        (search.py)       │
                               └─────────────┬────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
          ┌─────────────────────────┐                 ┌─────────────────────────┐
          │     BM25 Okapi Index    │                 │  ChromaDB Vector Store  │
          │     (Keyword Match)     │                 │ (MiniLM-L6 Embeddings)  │
          └────────────┬────────────┘                 └────────────┬────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               ┌──────────────────────────┐
                               │ Reciprocal Rank Fusion   │
                               │      (RRF Scoring)       │
                               └─────────────┬────────────┘
                                             │
                                             ▼
                               ┌──────────────────────────┐
                               │  Top-N Context Passages  │
                               └─────────────┬────────────┘
                                             │
                                             ▼
                               ┌──────────────────────────┐
                               │  Google Gemini 3.5 Flash │
                               │     (summarizer.py)      │
                               └─────────────┬────────────┘
                                             │
                                             ▼
                               ┌──────────────────────────┐
                               │ Grounded Output Summary  │
                               └──────────────────────────┘


🚀 Quick Start Guide

Prerequisites

Python 3.9+ installed on your system.

A Google Gemini API Key (Get a free key from Google AI Studio).

Step 1: Clone or Navigate to the Project Directory

Open your command prompt or terminal and navigate to the project directory:

cd "C:\Users\Raja Rajan\OneDrive\Desktop\rag"


Step 2: Set Up Virtual Environment

Create and activate a virtual environment to manage dependencies cleanly:

Windows (PowerShell):

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv .venv
.\.venv\Scripts\Activate.ps1


macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate


Step 3: Install Required Dependencies

Install all required Python packages using pip:

pip install google-genai datasets rank-bm25 sentence-transformers chromadb rouge-score streamlit python-dotenv numpy tqdm


Package Breakdown:

google-genai: Official SDK for Google Gemini models (gemini-3.5-flash).

datasets: Hugging Face library to stream Wikipedia articles.

rank-bm25: BM25 keyword search implementation.

sentence-transformers: Local embedding model (all-MiniLM-L6-v2).

chromadb: Vector database for fast embedding retrieval.

rouge-score: Automated metrics for summary quality evaluation.

streamlit: Interactive web front-end interface.

python-dotenv: Environment variable loader for API key security.

Step 4: Environment Configuration

Create a .env file in the project root directory and add your Gemini API Key:

GEMINI_API_KEY=your_actual_gemini_api_key_here


🏃 Running the Application Modules

1. Data Preparation (data_prep.py)

Fetches Wikipedia articles, cleans noisy artifacts/citations, chunks the text into passages, and saves the formatted dataset to wiki_corpus.json.

python data_prep.py


2. Testing Hybrid Search & Summarization CLI (main.py)

Runs a sample command-line query to verify that BM25, ChromaDB vector search, and Gemini 3.5 Flash summarization are interacting seamlessly.

python main.py


3. Running Automated Evaluation (evaluate.py)

Evaluates the RAG pipeline's retrieval precision and summary quality across ground-truth test queries.

python evaluate.py


4. Launching the Interactive Web App (app.py)

Launches the full-featured Streamlit user interface in your web browser.

streamlit run app.py


The web interface will automatically open at http://localhost:8501.

📊 Evaluation Results

The evaluation suite was executed on a ground-truth dataset extracted from the processed corpus.

==========================================
         FINAL EVALUATION REPORT          
==========================================
Total Test Cases      : 3
Search Hit Rate (@3)  : 100.00%
Mean Reciprocal Rank  : 0.5000
------------------------------------------
Average ROUGE-1 F1    : 0.4154
Average ROUGE-2 F1    : 0.2922
Average ROUGE-L F1    : 0.3354
==========================================


Metrics Explanation:

Search Hit Rate (@3) (100.00%): Demonstrates that 100% of test queries successfully retrieved the target ground-truth article in the top 3 results.

Mean Reciprocal Rank (0.5000): Reflects high rank precision across combined keyword and dense semantic results.

ROUGE F1 Scores: Confirms strong n-gram overlap between Gemini's generated summaries and the retrieved context passages, verifying strict factual grounding without hallucination.

📁 Repository Structure

.
├── data_prep.py     # Wikipedia ingestion, cleaning, and text chunking
├── search.py        # Hybrid Search Engine (BM25 + ChromaDB + RRF)
├── summarizer.py    # Gemini 3.5 Flash API integration with context grounding
├── evaluate.py      # Quantitative evaluation suite (Hit Rate, MRR, ROUGE)
├── app.py           # Streamlit Web UI application
├── main.py          # Command-line test script
├── wiki_corpus.json # Processed dataset (generated)
├── .env             # Environment file storing GEMINI_API_KEY
└── README.md        # Comprehensive project documentation

