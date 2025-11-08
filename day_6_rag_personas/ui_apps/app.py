"""
RAG System - Interactive Web UI
A beautiful Streamlit interface for your RAG implementation
"""

import streamlit as st
import lancedb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
import time

# Page configuration
st.set_page_config(
    page_title="RAG System - Persona Search",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .document-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the RAG system components"""
    try:
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Connect to LanceDB
        vector_store = LanceDBVectorStore(
            uri="../lancedb_data",
            table_name="personas_rag",
            mode="read"
        )
        
        # Initialize Ollama LLM
        llm = Ollama(
            model="gemma3:1b",
            base_url="http://localhost:11434",
            request_timeout=60.0
        )
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        # Create query engine
        query_engine = index.as_query_engine(
            llm=llm,
            response_mode="tree_summarize"
        )
        
        # Database connection for vector search
        db = lancedb.connect("../lancedb_data")
        
        return query_engine, embed_model, db, True
    except Exception as e:
        return None, None, None, str(e)

def perform_vector_search(query_text, embed_model, db, top_k=5):
    """Perform vector search on LanceDB"""
    try:
        table = db.open_table("personas_rag")
        query_embedding = embed_model.get_text_embedding(query_text)
        results = table.search(query_embedding).limit(top_k).to_pandas()
        return results
    except Exception as e:
        st.error(f"Vector search error: {e}")
        return None

def perform_rag_query(query_text, query_engine):
    """Perform RAG query with LLM"""
    try:
        response = query_engine.query(query_text)
        return str(response)
    except Exception as e:
        st.error(f"RAG query error: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🤖 RAG Persona Search System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about personas in the dataset</div>', unsafe_allow_html=True)
    
    # Initialize system
    with st.spinner("🔧 Initializing RAG system..."):
        query_engine, embed_model, db, status = initialize_rag_system()
    
    if status != True:
        st.error(f"❌ Failed to initialize RAG system: {status}")
        st.info("💡 Make sure you've run `python RAG_Implementation.py` first and Ollama is running!")
        return
    
    st.success("✅ RAG System Ready!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        search_mode = st.radio(
            "Search Mode",
            ["RAG with LLM", "Vector Search Only", "Both"],
            help="Choose how to process your query"
        )
        
        num_results = st.slider(
            "Number of Documents",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant documents to retrieve"
        )
        
        st.divider()
        
        st.header("📝 Sample Queries")
        sample_queries = [
            "Who are the AI experts?",
            "Find teachers and educators",
            "Tell me about climate scientists",
            "Who works in healthcare?",
            "Find data scientists",
            "Show me artists and designers"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}"):
                st.session_state.sample_query = query
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This RAG system uses:
        - **LlamaIndex** for orchestration
        - **LanceDB** for vector storage
        - **Ollama (gemma3:1b)** for generation
        - **HuggingFace** embeddings
        """)
        
        st.divider()
        
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle sample query from sidebar
    if "sample_query" in st.session_state:
        user_query = st.session_state.sample_query
        del st.session_state.sample_query
    else:
        # Chat input
        user_query = st.chat_input("💬 Ask a question about the personas...")
    
    if user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process query
        with st.chat_message("assistant"):
            response_container = st.container()
            
            with response_container:
                # Vector Search
                if search_mode in ["Vector Search Only", "Both"]:
                    with st.spinner("🔍 Searching vectors..."):
                        vector_results = perform_vector_search(user_query, embed_model, db, num_results)
                    
                    if vector_results is not None:
                        st.markdown("### 📄 Relevant Documents")
                        
                        for idx, row in vector_results.iterrows():
                            score = row.get('_distance', 'N/A')
                            text = row.get('text', 'N/A')
                            
                            if isinstance(score, (int, float)):
                                score_str = f"{score:.3f}"
                                # Color based on relevance
                                if score < 0.7:
                                    badge_color = "#28a745"  # green
                                elif score < 1.0:
                                    badge_color = "#ffc107"  # yellow
                                else:
                                    badge_color = "#dc3545"  # red
                            else:
                                score_str = str(score)
                                badge_color = "#6c757d"  # gray
                            
                            st.markdown(f"""
                            <div class="document-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <strong>Document {idx + 1}</strong>
                                    <span class="score-badge" style="background-color: {badge_color};">
                                        Score: {score_str}
                                    </span>
                                </div>
                                <div style="color: #333; line-height: 1.6;">
                                    {text[:400]}{'...' if len(text) > 400 else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # RAG with LLM
                if search_mode in ["RAG with LLM", "Both"]:
                    if search_mode == "Both":
                        st.markdown("---")
                    
                    st.markdown("### 🤖 AI Response")
                    
                    with st.spinner("🧠 Generating response..."):
                        start_time = time.time()
                        rag_response = perform_rag_query(user_query, query_engine)
                        elapsed_time = time.time() - start_time
                    
                    if rag_response:
                        st.markdown(f"""
                        <div class="result-box">
                            {rag_response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"⏱️ Response generated in {elapsed_time:.2f}s")
                
                # Save assistant response
                if search_mode == "RAG with LLM" and rag_response:
                    assistant_message = f"🤖 **AI Response:**\n\n{rag_response}"
                elif search_mode == "Vector Search Only" and vector_results is not None:
                    assistant_message = f"📄 Found {len(vector_results)} relevant documents"
                else:
                    assistant_message = "Response generated (see above)"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })

if __name__ == "__main__":
    main()



