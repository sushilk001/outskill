"""
Streamlit UI for Agentic RAG System
Interactive web interface for intelligent document Q&A with web search fallback
"""

import streamlit as st
import os
import logging
from agentic_rag_app import AgenticRAGSystem
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'pdfs_loaded' not in st.session_state:
        st.session_state.pdfs_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def sidebar_config():
    """Sidebar configuration and controls"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key Management
        st.subheader("🔑 API Keys")
        
        use_openrouter = st.checkbox(
            "Use OpenRouter",
            value=Config.USE_OPENROUTER,
            help="Use OpenRouter instead of OpenAI"
        )
        
        if use_openrouter:
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                value=Config.OPENROUTER_API_KEY,
                help="Get your key from https://openrouter.ai"
            )
            Config.OPENROUTER_API_KEY = api_key
            Config.USE_OPENROUTER = True
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=Config.OPENAI_API_KEY,
                help="Enter your OpenAI API key"
            )
            Config.OPENAI_API_KEY = api_key
            Config.USE_OPENROUTER = False
        
        st.divider()
        
        # Model Settings
        st.subheader("🤖 Model Settings")
        
        if use_openrouter:
            model_options = [
                "openai/gpt-4o",
                "openai/gpt-4-turbo",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro-1.5"
            ]
            selected_model = st.selectbox(
                "Model",
                model_options,
                index=0
            )
            Config.OPENROUTER_MODEL = selected_model
        else:
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
            selected_model = st.selectbox(
                "Model",
                model_options,
                index=0
            )
            Config.DEFAULT_MODEL = selected_model
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=Config.TEMPERATURE,
            step=0.1,
            help="Higher values make output more creative"
        )
        Config.TEMPERATURE = temperature
        
        st.divider()
        
        # RAG Settings
        st.subheader("📚 RAG Settings")
        
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=Config.TOP_K_RESULTS,
            help="How many document chunks to retrieve"
        )
        Config.TOP_K_RESULTS = top_k
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.error("❌ API Key missing")
        
        if st.session_state.pdfs_loaded:
            st.success("✅ PDFs loaded")
        else:
            st.warning("⚠️ No PDFs loaded")
        
        st.divider()
        
        # Actions
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Reset System"):
            st.session_state.rag_system = None
            st.session_state.pdfs_loaded = False
            st.session_state.chat_history = []
            st.rerun()


def initialize_rag_system():
    """Initialize the RAG system with error handling"""
    try:
        if st.session_state.rag_system is None:
            with st.spinner("🚀 Initializing RAG system..."):
                try:
                    st.session_state.rag_system = AgenticRAGSystem()
                    if st.session_state.rag_system.embeddings is None:
                        st.warning("⚠️ Embeddings failed to initialize")
                    return True
                except Exception as e:
                    logger.error(f"Error initializing RAG system: {e}", exc_info=True)
                    st.error(f"Error initializing system: {str(e)[:200]}")
                    return False
        
        # Reinitialize LLM if API key was updated
        api_key = Config.get_api_key()
        if api_key and api_key.strip():
            if (st.session_state.rag_system.llm is None or 
                (hasattr(st.session_state, 'last_api_key') and 
                 st.session_state.last_api_key != api_key)):
                try:
                    success = st.session_state.rag_system.reinitialize_llm()
                    if success:
                        st.session_state.last_api_key = api_key
                    else:
                        st.warning("⚠️ Could not initialize LLM. Check your API key.")
                except Exception as e:
                    logger.warning(f"Could not reinitialize LLM: {e}")
                    st.warning(f"⚠️ Could not reinitialize LLM: {str(e)[:100]}")
        
        return True
    except Exception as e:
        logger.error(f"Error in initialize_rag_system: {e}", exc_info=True)
        return False


def load_pdfs():
    """Load PDF documents into the system"""
    if not st.session_state.pdfs_loaded:
        pdf_dir = Config.PDF_DIR
        
        # Check if PDF directory exists
        if not os.path.exists(pdf_dir):
            st.warning(f"⚠️ PDF directory not found: {pdf_dir}")
            st.info("Creating directory. Please add PDF files to it.")
            os.makedirs(pdf_dir, exist_ok=True)
            return False
        
        # Check for PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            st.warning(f"⚠️ No PDF files found in: {pdf_dir}")
            st.info("Please add PDF files to the directory and reload.")
            return False
        
        # Load PDFs
        with st.spinner(f"📄 Loading {len(pdf_files)} PDF file(s)..."):
            try:
                num_chunks = st.session_state.rag_system.load_pdfs()
                st.session_state.pdfs_loaded = True
                st.success(f"✅ Loaded {num_chunks} chunks from {len(pdf_files)} PDF(s)")
                return True
            except Exception as e:
                st.error(f"❌ Error loading PDFs: {e}")
                return False
    return True


def chat_interface():
    """Main chat interface"""
    st.title("🤖 Agentic RAG System")
    st.markdown("### Intelligent Document Q&A with Web Search Fallback")
    
    # Check API key
    if not Config.get_api_key():
        st.error("❌ Please configure your API key in the sidebar")
        st.info("""
        **How to get started:**
        1. Enter your OpenAI or OpenRouter API key in the sidebar
        2. Add PDF files to the `data/pdfs/` directory
        3. Click 'Load PDFs' below
        4. Start asking questions!
        """)
        return
    
    # Initialize system
    if not initialize_rag_system():
        return
    
    # PDF Loading Section
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        pdf_dir = Config.PDF_DIR
        st.text(f"📁 PDF Directory: {pdf_dir}")
        
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            if pdf_files:
                st.text(f"📄 Found {len(pdf_files)} PDF file(s)")
                with st.expander("View PDF files"):
                    for pdf in pdf_files:
                        st.text(f"  • {pdf}")
            else:
                st.text("⚠️ No PDF files found")
    
    with col2:
        if st.button("📂 Load PDFs", type="primary", use_container_width=True):
            load_pdfs()
    
    with col3:
        if st.button("🔄 Reload PDFs", use_container_width=True):
            st.session_state.pdfs_loaded = False
            load_pdfs()
    
    st.divider()
    
    # System Flow Diagram
    with st.expander("📊 System Flow", expanded=False):
        st.markdown("""
        ```
        User Question
            ↓
        Retrieve from PDF
            ↓
        Check Relevance
            ↓
        ┌───────────┴───────────┐
        ↓                       ↓
    [RELEVANT]            [NOT RELEVANT]
        ↓                       ↓
    Generate Answer         Web Search
    from PDF                    ↓
        ↓                   Generate Answer
        ↓                   from Web
        └───────────┬───────────┘
                    ↓
                Return Answer
        ```
        
        **How it works:**
        1. 🔍 **Retrieve**: Search for relevant content in PDF documents
        2. 🤔 **Check**: Agent determines if PDF content is relevant
        3. ✅ **Route**: If relevant → use PDF; if not → search web
        4. 📝 **Generate**: Create answer from appropriate source
        5. ✨ **Return**: Deliver answer to user
        """)
    
    # Chat History Display
    st.subheader("💬 Chat")
    
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, entry in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(entry["question"])
            
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])
                
                # Show source
                source = entry.get("source", "unknown")
                if source == "pdf":
                    st.caption("📄 From PDF")
                elif source == "web":
                    st.caption("🌐 From Web")
                elif source == "error":
                    st.caption("⚠️ Error")
                
                # Show details expander
                with st.expander("📋 Details"):
                    st.json(entry.get("details", {}))
    
    # Chat Input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Check if PDFs are loaded
        if not st.session_state.pdfs_loaded:
            st.warning("⚠️ PDFs not loaded. System will only use web search.")
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Validate system is ready
                    if not st.session_state.rag_system:
                        st.error("System not initialized. Please refresh the page.")
                        return
                    
                    result = st.session_state.rag_system.answer_question(
                        question,
                        verbose=False
                    )
                    
                    if result.get("success"):
                        answer = result.get("answer", "No answer provided")
                        source = result.get("source", "unknown")
                        
                        st.markdown(answer)
                        
                        # Show source badge
                        if source == "pdf":
                            st.success("📄 Answer generated from PDF documents")
                        elif source == "web":
                            st.info("🌐 Answer generated from web search")
                        else:
                            st.warning(f"Answer source: {source}")
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "source": source,
                            "details": {
                                "relevance": result.get("relevance", {}),
                                "num_chunks": result.get("retrieval", {}).get("num_docs", 0)
                            }
                        })
                    else:
                        error_msg = result.get("answer", "Unknown error occurred")
                        st.error(f"❌ {error_msg}")
                        
                        # Save error to history for debugging
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": error_msg,
                            "source": "error",
                            "details": result
                        })
                
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)[:200]}"
                    st.error(f"❌ {error_msg}")
                    logger.error(f"Error in chat interface: {e}", exc_info=True)


def about_tab():
    """About page"""
    st.title("ℹ️ About Agentic RAG System")
    
    st.markdown("""
    ## 🤖 Agentic RAG System
    
    An intelligent document Q&A system that automatically routes questions to the most appropriate source.
    
    ### 🌟 Features
    
    - **📄 PDF Document Processing**: Load and index PDF documents for quick retrieval
    - **🤔 Intelligent Routing**: Agent-based decision making for source selection
    - **🔍 Semantic Search**: Find relevant information using embeddings
    - **🌐 Web Search Fallback**: Automatically searches web when PDF context is insufficient
    - **💬 Interactive Chat**: Natural conversation interface
    - **⚙️ Configurable**: Customize models, parameters, and behavior
    
    ### 🏗️ Architecture
    
    **Components:**
    1. **Document Loader**: Processes PDF files into searchable chunks
    2. **Vector Store**: Stores document embeddings for fast retrieval
    3. **Retriever**: Finds relevant document chunks for queries
    4. **Relevance Checker**: LLM-based agent that evaluates context quality
    5. **Answer Generator**: Creates responses from PDF or web sources
    6. **Web Search**: Fallback mechanism for out-of-scope questions
    
    **Flow:**
    ```
    Question → PDF Retrieval → Relevance Check → Route → Answer
                                     ↓
                              [PDF | Web Search]
    ```
    
    ### 🚀 Getting Started
    
    1. **Configure API Key**: Enter your OpenAI or OpenRouter API key in the sidebar
    2. **Add PDF Files**: Place PDF documents in the `data/pdfs/` directory
    3. **Load PDFs**: Click the "Load PDFs" button to process documents
    4. **Ask Questions**: Start chatting with your documents!
    
    ### 💡 Tips
    
    - **Document-specific questions** will be answered from PDFs
    - **General knowledge questions** will trigger web search
    - **Adjust temperature** for more creative or focused responses
    - **Increase chunks** (Top K) for more comprehensive context
    
    ### 🛠️ Tech Stack
    
    - **LangChain**: Framework for LLM applications
    - **OpenAI / OpenRouter**: Large language models
    - **FAISS**: Vector similarity search
    - **HuggingFace**: Sentence embeddings
    - **Streamlit**: Web interface
    - **PyPDF**: PDF document processing
    
    ### 📚 Resources
    
    - [LangChain Documentation](https://python.langchain.com/)
    - [OpenAI API](https://platform.openai.com/)
    - [OpenRouter](https://openrouter.ai/)
    - [FAISS](https://github.com/facebookresearch/faiss)
    
    ### 📝 Notes
    
    - Web search is currently simulated. To enable real web search:
      - Install: `pip install tavily-python` or `pip install google-search-results`
      - Get API key from Tavily or SerpAPI
      - Update the `web_search_simulation` method in `agentic_rag_app.py`
    
    ---
    
    **Created for educational purposes** | Part of the Outskill learning series
    """)


def main():
    """Main application"""
    init_session_state()
    
    # Sidebar
    sidebar_config()
    
    # Main content - Chat interface (no tabs, since chat_input can't be in tabs)
    chat_interface()
    
    # About section at the bottom
    st.divider()
    with st.expander("ℹ️ About", expanded=False):
        about_tab()


if __name__ == "__main__":
    main()

