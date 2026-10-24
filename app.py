import streamlit as st
from google.genai.errors import ServerError, APIError
from summarizer import GeminiSummarizer

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Research Assistant - Philosophy & Knowledge Base",
    page_icon="📚",
    layout="wide"
)

# 2. Sidebar Configuration Controls
st.sidebar.header("⚙️ Pipeline Settings")
length_option = st.sidebar.radio(
    "Select Summary Detail Level:",
    options=["short", "medium", "detailed"],
    index=1,
    format_func=lambda x: x.capitalize()
)

top_n_docs = st.sidebar.slider(
    "Number of Top Chunks to Retrieve:",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.markdown("---")

# Sample Questions Section in Sidebar
st.sidebar.header("💡 Try Sample Questions")
st.sidebar.caption("Click a pre-configured question to auto-fill:")

sample_questions = [
    "Select a question...",
    "What is alchemy?",
    "Who was Aristotle?",
    "What is altruism?",
    "What are the key concepts of ancient Greek philosophy?"
]

selected_sample = st.sidebar.selectbox("Sample Queries:", sample_questions)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by `google-genai` & `ChromaDB`")

# 3. Title & Corpus Topics Banner
st.title("📚 Corpus RAG Research Assistant")
st.markdown("""
Welcome! This application features an end-to-end **RAG (Retrieval-Augmented Generation) Pipeline** utilizing a **Hybrid Search Engine** 
(BM25 Keyword Search + Sentence Transformers Vector Embeddings fused via Reciprocal Rank Fusion) and **Google Gemini 3.5 Flash**.
""")

# Topic Badges / Corpus Summary Container
with st.expander("📖 **Explore Covered Corpus Topics & Scope**", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 🏛️ Classical Philosophy")
        st.markdown("- Aristotle & Metaphysics\n- Ancient Greek Intellectual History\n- Philosophical Frameworks")
    with col2:
        st.markdown("##### ⚗️ Science & History")
        st.markdown("- Alchemy & Early Chemistry\n- Paracelsus & Medicinal Knowledge\n- Scientific Method Roots")
    with col3:
        st.markdown("##### ⚖️ Ethics & Social Science")
        st.markdown("- Altruism & Moral Philosophy\n- Behavioral Science Concepts\n- Human Nature & Virtues")

st.divider()

# 4. Initialize Summarizer in Session State (Cached)
@st.cache_resource
def get_summarizer():
    return GeminiSummarizer()

try:
    summarizer = get_summarizer()
except Exception as e:
    st.error(f"Error initializing RAG Pipeline: {e}")
    st.info("Please verify that your `GEMINI_API_KEY` is set in your `.env` file or Streamlit Secrets.")
    st.stop()

# Handle Auto-filling query from sample selection
default_query_value = ""
if selected_sample != "Select a question...":
    default_query_value = selected_sample

# 5. User Query Input Interface
query = st.text_input(
    "Enter your research question:", 
    value=default_query_value,
    placeholder="e.g., What is alchemy? or Who was Aristotle?"
)

if st.button("Generate Summary", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid search query.")
    else:
        with st.spinner("Retrieving relevant context and generating summary via Gemini 3.5 Flash..."):
            try:
                # Execute RAG Pipeline
                summary, sources = summarizer.generate_summary(
                    query=query, 
                    top_n=top_n_docs, 
                    length_option=length_option
                )
                
                # Display Summary Result
                st.subheader("💡 Grounded Summary")
                st.write(summary)
                
                st.divider()
                
                # Display Retrieved Sources in an Expander Block
                st.subheader("🔍 Retrieved Context Sources")
                with st.expander("Click to view retrieved Wikipedia excerpts used for grounding", expanded=True):
                    for idx, doc in enumerate(sources):
                        st.markdown(f"### Source [{idx+1}]: {doc.get('title', 'Unknown Title')}")
                        st.caption(f"**Doc ID:** `{doc.get('doc_id', 'N/A')}`")
                        st.markdown(f"> {doc.get('content')}")
                        if idx < len(sources) - 1:
                            st.markdown("---")
                            
            except (ServerError, APIError) as api_err:
                # Friendly API busy/traffic notice for the user
                st.warning("⚠️ **Gemini API Traffic Alert:** The Gemini API server is currently experiencing temporary high demand or rate limits. Please wait a few seconds and click 'Generate Summary' again!")
                st.caption(f"Error Details: `{api_err}`")
                
            except Exception as err:
                st.error(f"An unexpected error occurred while generating response: {err}")