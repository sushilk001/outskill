"""
Setup Test Script
Quick test to verify your LangChain environment is configured correctly.
"""

import sys
import os


def test_imports():
    """Test if all required packages are installed"""
    print("\n🔍 Testing package imports...")
    
    packages = [
        ("langchain", "LangChain core"),
        ("langchain_openai", "LangChain OpenAI"),
        ("openai", "OpenAI SDK"),
        ("streamlit", "Streamlit"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
    ]
    
    missing = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed!")
    return True


def test_environment():
    """Test if environment variables are set"""
    print("\n🔍 Testing environment configuration...")
    
    from config import Config
    
    if not Config.OPENAI_API_KEY:
        print("  ❌ OPENAI_API_KEY not set")
        print("\n⚠️  Please create a .env file with your API key:")
        print("     echo 'OPENAI_API_KEY=your_key_here' > .env")
        return False
    
    print(f"  ✅ OPENAI_API_KEY is set")
    print(f"  ✅ Model: {Config.DEFAULT_MODEL}")
    print(f"  ✅ Temperature: {Config.TEMPERATURE}")
    
    return True


def test_directories():
    """Test if required directories exist"""
    print("\n🔍 Testing directory structure...")
    
    from config import Config
    
    dirs = [
        (Config.DATA_DIR, "Data directory"),
        (Config.VECTOR_STORE_DIR, "Vector store directory"),
    ]
    
    for path, name in dirs:
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ℹ️  {name} will be created when needed: {path}")
    
    return True


def test_api_connection():
    """Test connection to OpenAI API"""
    print("\n🔍 Testing OpenAI API connection...")
    
    from config import Config
    
    if not Config.OPENAI_API_KEY:
        print("  ⏭️  Skipped (no API key)")
        return True
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=Config.OPENAI_API_KEY,
            max_tokens=50
        )
        
        print("  🔄 Sending test request...")
        response = llm.invoke("Say 'Hello from LangChain!' and nothing else.")
        
        print(f"  ✅ API connection successful!")
        print(f"  💬 Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"  ❌ API connection failed: {e}")
        print("\n⚠️  Possible issues:")
        print("     - Invalid API key")
        print("     - No credits in account")
        print("     - Network issues")
        return False


def test_embeddings():
    """Test embedding model"""
    print("\n🔍 Testing embedding model...")
    
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from config import Config
        
        print(f"  🔄 Loading model: {Config.EMBEDDING_MODEL}")
        print("     (First run may take a moment to download...)")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Test embedding
        test_text = "Hello, this is a test."
        vector = embeddings.embed_query(test_text)
        
        print(f"  ✅ Embeddings working! (Vector size: {len(vector)})")
        return True
        
    except Exception as e:
        print(f"  ❌ Embeddings test failed: {e}")
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\n📚 Next steps:")
        print("   1. Read QUICKSTART.md for usage guide")
        print("   2. Try: python basic_chain.py")
        print("   3. Or launch UI: streamlit run streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("   - Install packages: pip install -r requirements.txt")
        print("   - Set API key: echo 'OPENAI_API_KEY=your_key' > .env")
        print("   - Check your OpenAI account has credits")
    
    print("="*60)


def main():
    """Run all tests"""
    print("="*60)
    print("🔧 LangChain Setup Test")
    print("="*60)
    print("\nThis script will verify your environment is configured correctly.")
    
    results = {}
    
    # Run tests
    results["Package Imports"] = test_imports()
    results["Environment Config"] = test_environment()
    results["Directory Structure"] = test_directories()
    
    # Only test API if basic setup is OK
    if results["Package Imports"] and results["Environment Config"]:
        results["API Connection"] = test_api_connection()
        results["Embedding Model"] = test_embeddings()
    else:
        print("\n⏭️  Skipping API and embedding tests (fix basic setup first)")
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

