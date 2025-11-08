"""
Quick diagnostic script to test LLM integration
"""

import sys
print("Testing LLM Integration...")
print("=" * 60)

# Test 1: Check Ollama connection
print("\n1. Testing Ollama Connection...")
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("   ✅ Ollama is accessible")
    else:
        print(f"   ❌ Ollama returned status code: {response.status_code}")
except Exception as e:
    print(f"   ❌ Cannot connect to Ollama: {e}")
    sys.exit(1)

# Test 2: Check if gemma3:1b is available
print("\n2. Checking gemma3:1b model...")
try:
    data = response.json()
    models = [m['name'] for m in data.get('models', [])]
    if 'gemma3:1b' in models:
        print("   ✅ gemma3:1b model is available")
    else:
        print(f"   ❌ gemma3:1b not found. Available models: {models}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error checking models: {e}")
    sys.exit(1)

# Test 3: Test LlamaIndex Ollama integration
print("\n3. Testing LlamaIndex Ollama integration...")
try:
    from llama_index.llms.ollama import Ollama
    
    llm = Ollama(
        model="gemma3:1b",
        base_url="http://localhost:11434",
        request_timeout=60.0
    )
    print("   ✅ Ollama LLM initialized")
    
    # Test a simple completion
    print("\n4. Testing LLM generation...")
    response = llm.complete("Say hello in one sentence.")
    print(f"   ✅ LLM Response: {response.text[:100]}")
    
except Exception as e:
    print(f"   ❌ LlamaIndex integration error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test with RAG system
print("\n5. Testing RAG System...")
try:
    import lancedb
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.lancedb import LanceDBVectorStore
    from llama_index.core import VectorStoreIndex
    
    # Initialize components
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("   ✅ Embedding model loaded")
    
    vector_store = LanceDBVectorStore(
        uri="../lancedb_data",
        table_name="personas_rag",
        mode="read"
    )
    print("   ✅ Vector store connected")
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    print("   ✅ Index created")
    
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize"
    )
    print("   ✅ Query engine created")
    
    # Test a query
    print("\n6. Testing RAG Query...")
    test_query = "Who are the technology experts?"
    print(f"   Query: {test_query}")
    response = query_engine.query(test_query)
    print(f"   ✅ Response: {response}")
    
except Exception as e:
    print(f"   ❌ RAG system error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED! LLM integration is working correctly.")
print("=" * 60)



