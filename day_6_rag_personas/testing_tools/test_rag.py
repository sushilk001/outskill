#!/usr/bin/env python3
"""
Interactive RAG Testing Script
Run this to test your RAG system with custom queries
"""

import os
import lancedb
import asyncio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def setup_rag_system():
    """Initialize the RAG system"""
    print("🔧 Initializing RAG system...")
    
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    # Connect to existing LanceDB
    vector_store = LanceDBVectorStore(
        uri="../lancedb_data",
        table_name="personas_rag",
        mode="read"  # Read existing data
    )
    
    # Initialize local Ollama LLM
    llm = Ollama(
        model="gemma3:1b",
        base_url="http://localhost:11434",
        request_timeout=60.0
    )
    
    # Create index and query engine
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize"
    )
    
    print("✅ RAG system ready!\n")
    return query_engine, embed_model

def test_vector_search_only(query_text, embed_model, top_k=3):
    """Perform vector search without LLM"""
    print(f"\n🔍 Vector Search Results (Top {top_k}):")
    print("-" * 60)
    
    # Connect to LanceDB
    db = lancedb.connect("../lancedb_data")
    table = db.open_table("personas_rag")
    
    # Get query embedding and search
    query_embedding = embed_model.get_text_embedding(query_text)
    results = table.search(query_embedding).limit(top_k).to_pandas()
    
    for idx, row in results.iterrows():
        score = row.get('_distance', 'N/A')
        text = row.get('text', 'N/A')
        
        if isinstance(score, (int, float)):
            score_str = f"{score:.3f}"
        else:
            score_str = str(score)
        
        print(f"\n📄 Result {idx + 1} (Similarity Score: {score_str}):")
        print(f"{text[:300]}{'...' if len(text) > 300 else ''}")
    print()

def test_rag_with_llm(query_text, query_engine):
    """Test RAG with LLM generation"""
    print("\n🤖 RAG with LLM Response:")
    print("-" * 60)
    
    response = query_engine.query(query_text)
    print(f"\n{response}\n")

def run_sample_tests(query_engine, embed_model):
    """Run some sample test queries"""
    print("\n" + "="*60)
    print("📝 SAMPLE TEST QUERIES")
    print("="*60)
    
    sample_queries = [
        "Who are the technology experts in the database?",
        "Find me teachers or educators",
        "Tell me about people working on climate change"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print('='*60)
        
        # Vector search
        test_vector_search_only(query, embed_model, top_k=2)
        
        # RAG with LLM
        test_rag_with_llm(query, query_engine)

def run_interactive_mode(query_engine, embed_model):
    """Interactive mode for custom queries"""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE MODE")
    print("="*60)
    print("\nEnter your queries to test the RAG system.")
    print("Commands:")
    print("  - Type your question to get RAG response")
    print("  - Type 'vector <query>' for vector search only")
    print("  - Type 'both <query>' for both vector search + RAG")
    print("  - Type 'samples' to run sample tests")
    print("  - Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            user_input = input("💬 Your query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'samples':
                run_sample_tests(query_engine, embed_model)
                continue
            
            if user_input.lower().startswith('vector '):
                query = user_input[7:].strip()
                test_vector_search_only(query, embed_model, top_k=3)
            
            elif user_input.lower().startswith('both '):
                query = user_input[5:].strip()
                test_vector_search_only(query, embed_model, top_k=3)
                test_rag_with_llm(query, query_engine)
            
            else:
                # Default: RAG with LLM
                test_rag_with_llm(user_input, query_engine)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 RAG SYSTEM TESTER")
    print("="*60)
    print("\nThis script allows you to test your RAG implementation")
    print("with custom queries using the gemma3:1b model.\n")
    
    try:
        # Setup RAG system
        query_engine, embed_model = setup_rag_system()
        
        # Ask user what they want to do
        print("What would you like to do?")
        print("  1. Run sample tests")
        print("  2. Interactive mode (enter your own queries)")
        print("  3. Both")
        
        choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip() or "2"
        
        if choice == "1":
            run_sample_tests(query_engine, embed_model)
        elif choice == "3":
            run_sample_tests(query_engine, embed_model)
            run_interactive_mode(query_engine, embed_model)
        else:
            run_interactive_mode(query_engine, embed_model)
    
    except FileNotFoundError:
        print("\n❌ Error: LanceDB data not found!")
        print("Please run 'python RAG_Implementation.py' first to create the database.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. You've run RAG_Implementation.py first")
        print("  2. Ollama is running (try: ollama serve)")
        print("  3. gemma3:1b model is available (try: ollama list)\n")

if __name__ == "__main__":
    main()



