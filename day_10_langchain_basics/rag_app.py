"""
RAG (Retrieval Augmented Generation) Application
Demonstrates document loading, vector storage, and retrieval-based Q&A.
"""

import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import Config


def create_sample_documents():
    """Create sample documents for demonstration"""
    print("\n📄 Creating sample documents...")
    
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    documents = {
        "python_basics.txt": """
Python is a high-level, interpreted programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.

Key Features of Python:
1. Easy to learn and read
2. Versatile - used for web development, data science, AI, automation
3. Large standard library
4. Active community and extensive third-party packages

Python uses indentation to define code blocks, making it visually clean.
Common data types include integers, floats, strings, lists, tuples, and dictionaries.
""",
        
        "data_science.txt": """
Data Science in Python:

Python has become the leading language for data science due to powerful libraries:

- NumPy: Numerical computing with arrays and matrices
- Pandas: Data manipulation and analysis with DataFrames
- Matplotlib/Seaborn: Data visualization
- Scikit-learn: Machine learning algorithms
- TensorFlow/PyTorch: Deep learning frameworks

A typical data science workflow includes:
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Building
6. Model Evaluation
7. Deployment
""",
        
        "web_development.txt": """
Web Development with Python:

Popular Python web frameworks:

Django:
- Full-featured web framework
- Includes ORM, admin interface, authentication
- "Batteries included" philosophy
- Great for large applications

Flask:
- Lightweight micro-framework
- Minimal and flexible
- Easy to get started
- Good for small to medium applications

FastAPI:
- Modern, fast framework
- Automatic API documentation
- Type hints and validation
- Async support

Python can also be used for web scraping with BeautifulSoup and Scrapy.
"""
    }
    
    for filename, content in documents.items():
        filepath = os.path.join(Config.DATA_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(content.strip())
    
    print(f"✅ Created {len(documents)} sample documents in {Config.DATA_DIR}")
    return Config.DATA_DIR


def load_and_split_documents(data_dir):
    """Load documents and split into chunks"""
    print("\n📚 Loading documents...")
    
    # Load all text files from directory
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
    
    # Split documents into chunks
    print("\n✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    return chunks


def create_vector_store(chunks):
    """Create vector store from document chunks"""
    print("\n🔢 Creating embeddings and vector store...")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created")
    
    return vector_store


def create_qa_chain(vector_store):
    """Create a QA chain with the vector store"""
    print("\n🔗 Creating QA chain...")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.3,  # Lower temperature for factual responses
        api_key=Config.OPENAI_API_KEY
    )
    
    # Custom prompt template
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✅ QA chain created")
    return qa_chain


def ask_question(qa_chain, question):
    """Ask a question and get an answer with sources"""
    print("\n" + "="*60)
    print(f"❓ Question: {question}")
    print("="*60)
    
    result = qa_chain.invoke({"query": question})
    
    print(f"\n💡 Answer:\n{result['result']}")
    
    print("\n📖 Sources:")
    for i, doc in enumerate(result['source_documents'], 1):
        source = doc.metadata.get('source', 'Unknown')
        print(f"\n{i}. {source}")
        print(f"   Preview: {doc.page_content[:150]}...")


def run_rag_demo():
    """Run complete RAG demonstration"""
    print("\n" + "="*60)
    print("🚀 RAG (Retrieval Augmented Generation) Demo")
    print("="*60)
    
    # Step 1: Create sample documents
    data_dir = create_sample_documents()
    
    # Step 2: Load and split documents
    chunks = load_and_split_documents(data_dir)
    
    # Step 3: Create vector store
    vector_store = create_vector_store(chunks)
    
    # Step 4: Create QA chain
    qa_chain = create_qa_chain(vector_store)
    
    # Step 5: Ask questions
    questions = [
        "What are the key features of Python?",
        "Which libraries are used for data science in Python?",
        "What is the difference between Django and Flask?",
        "How many steps are in a typical data science workflow?"
    ]
    
    for question in questions:
        ask_question(qa_chain, question)
    
    return qa_chain


def interactive_qa(qa_chain):
    """Interactive Q&A session"""
    print("\n" + "="*60)
    print("💬 Interactive Q&A Mode (type 'quit' to exit)")
    print("="*60)
    print("\nAsk questions about the documents!\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! 👋")
            break
        
        if not question:
            continue
        
        try:
            result = qa_chain.invoke({"query": question})
            print(f"\n💡 Answer: {result['result']}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main function"""
    if not Config.OPENAI_API_KEY:
        print("\n❌ Error: OPENAI_API_KEY not set!")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    try:
        # Run demo
        qa_chain = run_rag_demo()
        
        # Ask if user wants interactive mode
        print("\n" + "="*60)
        choice = input("\nWould you like to try interactive Q&A? (yes/no): ").strip().lower()
        
        if choice in ['yes', 'y']:
            interactive_qa(qa_chain)
        
        print("\n✅ RAG demo completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

