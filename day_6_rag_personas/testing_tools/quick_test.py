#!/usr/bin/env python3
"""
Quick RAG Test - Pass your query as a command line argument
Usage: python quick_test.py "your query here"
"""

import sys
import lancedb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

def quick_rag_query(query_text):
    """Perform a quick RAG query"""
    print(f"\n🔍 Query: {query_text}\n")
    
    # Initialize components
    print("Initializing RAG system...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    vector_store = LanceDBVectorStore(
        uri="../lancedb_data",
        table_name="personas_rag",
        mode="read"
    )
    
    llm = Ollama(
        model="gemma3:1b",
        base_url="http://localhost:11434",
        request_timeout=60.0
    )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize"
    )
    
    # Get response
    print("Getting response...\n")
    response = query_engine.query(query_text)
    
    print("="*60)
    print("🤖 RAG Response:")
    print("="*60)
    print(f"\n{response}\n")
    
    # Also show top 3 relevant documents
    print("="*60)
    print("📄 Top 3 Relevant Documents:")
    print("="*60)
    
    db = lancedb.connect("../lancedb_data")
    table = db.open_table("personas_rag")
    query_embedding = embed_model.get_text_embedding(query_text)
    results = table.search(query_embedding).limit(3).to_pandas()
    
    for idx, row in results.iterrows():
        score = row.get('_distance', 'N/A')
        text = row.get('text', 'N/A')
        print(f"\n{idx + 1}. (Score: {score:.3f})")
        print(f"   {text[:200]}...")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Please provide a query!")
        print("Usage: python quick_test.py \"your query here\"\n")
        print("Examples:")
        print("  python quick_test.py \"Who are the AI experts?\"")
        print("  python quick_test.py \"Find teachers in the dataset\"")
        print("  python quick_test.py \"Tell me about environmental scientists\"\n")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    try:
        quick_rag_query(query)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. You've run RAG_Implementation.py first")
        print("  2. Ollama is running")
        print("  3. gemma3:1b model is available\n")
        sys.exit(1)



