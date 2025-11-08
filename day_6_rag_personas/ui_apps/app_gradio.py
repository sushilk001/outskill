"""
RAG System - Gradio Interface
A simple and elegant interface for your RAG implementation
"""

import gradio as gr
import lancedb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

# Initialize RAG system
print("🔧 Initializing RAG system...")

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
    
    # Database connection
    db = lancedb.connect("../lancedb_data")
    
    print("✅ RAG system initialized successfully!")
    SYSTEM_READY = True
    
except Exception as e:
    print(f"❌ Error initializing system: {e}")
    SYSTEM_READY = False
    error_message = str(e)

def format_documents(results):
    """Format vector search results"""
    output = ""
    for idx, row in results.iterrows():
        score = row.get('_distance', 'N/A')
        text = row.get('text', 'N/A')
        
        if isinstance(score, (int, float)):
            score_str = f"{score:.3f}"
            if score < 0.7:
                relevance = "🟢 Highly Relevant"
            elif score < 1.0:
                relevance = "🟡 Moderately Relevant"
            else:
                relevance = "🔴 Less Relevant"
        else:
            score_str = str(score)
            relevance = "⚪ Unknown"
        
        output += f"""
### 📄 Document {idx + 1}
**Relevance:** {relevance} (Score: {score_str})

{text[:400]}{'...' if len(text) > 400 else ''}

---
"""
    return output

def vector_search_only(query, num_results):
    """Perform vector search without LLM"""
    if not SYSTEM_READY:
        return f"❌ System not ready: {error_message}"
    
    try:
        table = db.open_table("personas_rag")
        query_embedding = embed_model.get_text_embedding(query)
        results = table.search(query_embedding).limit(int(num_results)).to_pandas()
        
        output = f"# 🔍 Vector Search Results\n\n"
        output += f"Found **{len(results)}** relevant documents:\n\n"
        output += format_documents(results)
        
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"

def rag_with_llm(query, num_results):
    """Perform RAG query with LLM"""
    if not SYSTEM_READY:
        return f"❌ System not ready: {error_message}", ""
    
    try:
        # Get RAG response
        response = query_engine.query(query)
        rag_output = f"# 🤖 AI Response\n\n{response}\n\n"
        
        # Get supporting documents
        table = db.open_table("personas_rag")
        query_embedding = embed_model.get_text_embedding(query)
        results = table.search(query_embedding).limit(int(num_results)).to_pandas()
        
        docs_output = f"# 📚 Supporting Documents\n\n"
        docs_output += format_documents(results)
        
        return rag_output, docs_output
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def combined_search(query, num_results, mode):
    """Handle different search modes"""
    if mode == "Vector Search Only":
        result = vector_search_only(query, num_results)
        return result, ""
    elif mode == "RAG with LLM":
        return rag_with_llm(query, num_results)
    else:  # Both
        rag_resp, docs = rag_with_llm(query, num_results)
        return rag_resp, docs

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Persona Search") as demo:
    gr.Markdown("""
    # 🤖 RAG Persona Search System
    ### Ask questions about personas in the dataset
    """)
    
    if not SYSTEM_READY:
        gr.Markdown(f"""
        ⚠️ **System Not Ready**
        
        Error: {error_message}
        
        Please make sure:
        1. You've run `python RAG_Implementation.py` first
        2. Ollama is running
        3. The `lancedb_data/` directory exists
        """)
    else:
        gr.Markdown("✅ **System Ready!** Start asking questions below.")
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="💬 Your Question",
                placeholder="e.g., Who are the AI experts in the dataset?",
                lines=2
            )
            
            with gr.Row():
                search_button = gr.Button("🔍 Search", variant="primary", scale=2)
                clear_button = gr.Button("🗑️ Clear", scale=1)
            
        with gr.Column(scale=1):
            mode_selector = gr.Radio(
                choices=["Vector Search Only", "RAG with LLM", "Both"],
                value="RAG with LLM",
                label="🎯 Search Mode",
                info="Choose how to process your query"
            )
            
            num_results = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="📊 Number of Documents",
                info="How many documents to retrieve"
            )
    
    # Sample queries
    gr.Markdown("### 📝 Sample Queries (Click to use)")
    with gr.Row():
        sample1 = gr.Button("👨‍💻 AI Experts", size="sm")
        sample2 = gr.Button("👨‍🏫 Teachers", size="sm")
        sample3 = gr.Button("🌍 Climate Scientists", size="sm")
        sample4 = gr.Button("🏥 Healthcare", size="sm")
        sample5 = gr.Button("📊 Data Scientists", size="sm")
    
    # Output areas
    with gr.Row():
        with gr.Column():
            response_output = gr.Markdown(label="Response")
        
        with gr.Column():
            documents_output = gr.Markdown(label="Documents")
    
    # Event handlers
    def process_query(query, num_results, mode):
        if not query.strip():
            return "⚠️ Please enter a question", ""
        return combined_search(query, num_results, mode)
    
    search_button.click(
        fn=process_query,
        inputs=[query_input, num_results, mode_selector],
        outputs=[response_output, documents_output]
    )
    
    clear_button.click(
        fn=lambda: ("", "", ""),
        outputs=[query_input, response_output, documents_output]
    )
    
    # Sample query buttons
    sample1.click(lambda: "Who are the artificial intelligence experts?", outputs=query_input)
    sample2.click(lambda: "Find teachers and educators", outputs=query_input)
    sample3.click(lambda: "Tell me about climate scientists", outputs=query_input)
    sample4.click(lambda: "Who works in healthcare?", outputs=query_input)
    sample5.click(lambda: "Find data scientists", outputs=query_input)
    
    gr.Markdown("""
    ---
    ### ℹ️ About
    This RAG system uses:
    - **LlamaIndex** for orchestration
    - **LanceDB** for vector storage  
    - **Ollama (gemma3:1b)** for generation
    - **HuggingFace** embeddings (BAAI/bge-small-en-v1.5)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )



