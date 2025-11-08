# 📚 RAG Implementation - Code Walkthrough & Execution Flow

## 🎯 Overview
This code implements a complete RAG (Retrieval Augmented Generation) system using LlamaIndex and LanceDB with three different LLM options.

---

## 🔄 Execution Flow Diagram

```
START
  ↓
[1] Import Libraries (Lines 24-47)
  ↓
[2] Load Data from HuggingFace (Lines 51-84)
  ↓
[3] Setup LanceDB Database (Lines 86-101)
  ↓
[4] Run main() async function (Lines 139-150)
  ↓
  ├─→ [5] Create Embeddings & Index (Lines 105-137)
  ↓
  ├─→ [6] Test Vector Search (Lines 168-201)
  ↓
  └─→ [7] Show Usage Examples (Lines 492-513)
  ↓
END
```

---

## 📝 Step-by-Step Code Execution

### **STEP 1: Import Libraries** (Lines 24-49)

```python
import os, lancedb, subprocess, requests, time
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.ollama import Ollama
```

**What happens:**
- Imports all necessary libraries
- Sets up async support with `nest_asyncio`
- Prints "All libraries imported successfully"

**Purpose:** Load all dependencies needed for RAG system

---

### **STEP 2: Data Preparation** (Lines 51-84)

```python
def prepare_data(num_samples=100):
    # Load dataset from HuggingFace
    dataset = load_dataset("dvilasuero/finepersonas-v0.1-tiny", split="train")
    
    # Create Document objects
    documents = []
    for i, persona in enumerate(dataset.select(range(num_samples))):
        doc = Document(
            text=persona["persona"],
            metadata={"persona_id": i, "source": "finepersonas-dataset"}
        )
        documents.append(doc)
    
    return documents

# Execute: Load 100 personas
documents = prepare_data(num_samples=100)
```

**What happens:**
1. Downloads "finepersonas" dataset from HuggingFace (5000 personas total)
2. Selects first 100 personas
3. Creates `Document` objects with text and metadata
4. Saves each persona to `data/persona_X.txt` files
5. Returns list of 100 Document objects

**Output:** 
```
Loading 100 personas from dataset...
Prepared 100 documents
```

**Result:** `documents` = List of 100 Document objects

---

### **STEP 3: Database Setup** (Lines 86-101)

```python
def setup_lancedb_store(table_name="personas_rag"):
    db = lancedb.connect("./lancedb_data")
    return db, table_name

# Execute: Connect to database
db, table_name = setup_lancedb_store()
```

**What happens:**
1. Creates/connects to LanceDB at `./lancedb_data/`
2. Prepares table name "personas_rag"
3. Returns database connection and table name

**Output:**
```
Setting up LanceDB connection...
Connected to LanceDB, table: personas_rag
```

**Result:** `db` = Database connection, `table_name` = "personas_rag"

---

### **STEP 4: Main Function Entry** (Lines 539-540)

```python
if __name__ == "__main__":
    asyncio.run(main())
```

**What happens:**
- Python reaches the end of the file
- Checks if script is run directly (not imported)
- Executes `main()` async function

**This triggers the async workflow ↓**

---

### **STEP 5: Create Embeddings & Vector Index** (Lines 105-144)

```python
async def create_and_populate_index(documents, db, table_name):
    # 5.1: Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    # 5.2: Create vector store
    vector_store = LanceDBVectorStore(
        uri="./lancedb_data",
        table_name=table_name,
        mode="overwrite"
    )
    
    # 5.3: Create ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=20),
            embed_model,
        ],
        vector_store=vector_store,
    )
    
    # 5.4: Process documents
    nodes = await pipeline.arun(documents=documents)
    
    return vector_store, embed_model
```

**What happens in detail:**

**5.1: Embedding Model (Line 112-114)**
- Downloads `BAAI/bge-small-en-v1.5` model from HuggingFace
- This model converts text → 384-dimensional vectors
- Model size: ~130MB

**5.2: Vector Store (Lines 117-121)**
- Creates LanceDB vector store at `./lancedb_data/personas_rag.lance`
- Mode: "overwrite" (deletes existing data)
- Prepares to store vectors

**5.3: Ingestion Pipeline (Lines 124-130)**
- **SentenceSplitter**: Breaks documents into 512-character chunks (20 char overlap)
- **embed_model**: Converts each chunk into vector embeddings
- **vector_store**: Stores vectors in LanceDB

**5.4: Process Documents (Line 134)**
- Takes 100 documents
- Splits into ~100 chunks (depending on text length)
- Generates embeddings for each chunk
- Stores in LanceDB

**Output:**
```
Creating embedding model and ingestion pipeline...
Processing documents and creating embeddings...
Successfully processed 100 text chunks
```

**Result:** 
- `vector_store` = LanceDB store with 100 embedded chunks
- `embed_model` = Embedding model for future queries

---

### **STEP 6: Vector Search Test** (Lines 168-201)

```python
def test_vector_search():
    queries = [
        "technology and artificial intelligence expert",
        "teacher educator professor",
        "environment climate sustainability", 
        "art culture heritage creative"
    ]
    
    for query in queries:
        # Convert query to vector
        query_embedding = embed_model.get_text_embedding(query)
        
        # Search LanceDB
        table = db.open_table(table_name)
        results = table.search(query_embedding).limit(3).to_pandas()
        
        # Display results
        for idx, row in results.iterrows():
            print(f"Result {idx + 1} (Score: {row['_distance']}):")
            print(f"{row['text'][:200]}...")
```

**What happens:**

1. **For each query:**
   - Convert query text → vector (384 dimensions)
   - Search LanceDB for similar vectors
   - Return top 3 most similar chunks

2. **Similarity Calculation:**
   - Uses cosine distance
   - Lower score = more similar
   - Returns: document text + distance score

**Output Example:**
```
Query: technology and artificial intelligence expert
------------------------------
Result 1 (Score: 0.589):
A computer scientist or electronics engineer researching...

Result 2 (Score: 0.626):
An aerospace engineer or astrobiologist interested in...
```

**Result:** Prints 4 queries × 3 results = 12 relevant documents

---

### **STEP 7: Show Usage Examples** (Lines 492-513)

```python
def show_usage_examples():
    print("Usage Examples:")
    print("1. Vector Search Only: test_vector_search()")
    print("2. HuggingFace API RAG: await test_huggingface_rag()")
    print("3. Local LLM RAG: await test_local_llm_rag()")
```

**What happens:**
- Displays how to use the three different RAG approaches
- Educational output for the user

---

## 🔍 Key Components Explained

### **1. Document Object** (Lines 67-73)

```python
doc = Document(
    text="A software engineer...",
    metadata={"persona_id": 0, "source": "finepersonas-dataset"}
)
```

**Contains:**
- `text`: The actual persona description
- `metadata`: Additional info (ID, source)

---

### **2. Embedding Model** (Lines 112-114)

```python
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

**What it does:**
- Converts text → numerical vectors
- "software engineer" → [0.23, -0.45, 0.12, ..., 0.89] (384 numbers)
- Similar texts have similar vectors

**Example:**
```
"AI expert"          → [0.8, 0.3, -0.1, ...]
"machine learning"   → [0.7, 0.4, -0.2, ...]  ← Close!
"banana recipe"      → [-0.2, -0.9, 0.7, ...]  ← Far away
```

---

### **3. SentenceSplitter** (Line 126)

```python
SentenceSplitter(chunk_size=512, chunk_overlap=20)
```

**Why split documents?**
- Original doc: 2000 characters
- Too long for embeddings → loses context
- Solution: Break into 512-char chunks

**Example:**
```
Original: "John is a data scientist with 10 years experience..."
          (2000 characters)

Splits into:
Chunk 1: "John is a data scientist with 10 years..." (512 chars)
Chunk 2: "...10 years experience in machine learning..." (512 chars)
         ↑ 20 char overlap
Chunk 3: "...machine learning and built AI models..." (512 chars)
```

---

### **4. Vector Search** (Lines 159-165)

```python
query_embedding = embed_model.get_text_embedding(query)
results = table.search(query_embedding).limit(3)
```

**How it works:**

1. **Query:** "Who are the AI experts?"
2. **Convert to vector:** [0.8, 0.3, -0.1, ..., 0.5]
3. **Search database:**
   ```
   Compare query vector to all 100 stored vectors
   Calculate similarity (cosine distance)
   Sort by similarity
   Return top 3
   ```

4. **Result:** 3 most relevant persona chunks

---

### **5. RAG Query Engine** (Lines 210-231)

```python
def create_query_engine(vector_store, embed_model, llm):
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model)
    query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")
    return query_engine
```

**What it does:**

**Without LLM (Vector Search Only):**
```
Query → Vector → Search → Return matching docs
```

**With LLM (RAG):**
```
Query → Vector → Search → Get top docs → Send to LLM → Generate answer
```

**Example:**
```
Query: "Who are the AI experts?"

Vector Search returns:
- Doc 1: "Computer scientist working on neural networks..."
- Doc 2: "Data scientist specializing in deep learning..."
- Doc 3: "ML engineer with 5 years experience..."

LLM receives context + question:
"Given these documents: [Doc 1, Doc 2, Doc 3]
 Answer: Who are the AI experts?"

LLM generates:
"The dataset contains computer scientists and data scientists 
 specializing in neural networks and deep learning..."
```

---

## 🎨 Three LLM Options

### **Option 1: Vector Search Only** (Lines 155-201)
```python
results = perform_vector_search(db, table_name, query, embed_model)
```
- **Speed:** ⚡ Very fast (<1 second)
- **Quality:** Raw documents, no generation
- **Cost:** Free
- **Use case:** Quick lookups

---

### **Option 2: HuggingFace API** (Lines 240-276)
```python
llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-beta")
query_engine = create_query_engine(vector_store, embed_model, llm)
response = query_engine.query("Who are the AI experts?")
```
- **Speed:** ~5-10 seconds
- **Quality:** High-quality generated responses
- **Cost:** API costs (but has free tier)
- **Use case:** Production apps

---

### **Option 3: Local LLM (Ollama)** (Lines 360-398)
```python
llm = Ollama(model="gemma3:1b", base_url="http://localhost:11434")
query_engine = create_query_engine(vector_store, embed_model, llm)
response = query_engine.query("Who are the AI experts?")
```
- **Speed:** ~3-5 seconds
- **Quality:** Good generated responses
- **Cost:** Free (runs locally)
- **Use case:** Privacy-focused, offline apps

---

## 📊 Data Flow Visualization

```
USER QUERY: "Who are the AI experts?"
    ↓
┌─────────────────────────────────────────┐
│ 1. EMBEDDING                            │
│    Text → Vector [0.8, 0.3, -0.1, ...]  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. VECTOR SEARCH (LanceDB)              │
│    Compare to 100 stored vectors        │
│    Find top 3 most similar              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. RETRIEVED DOCUMENTS                  │
│    Doc 1: "Computer scientist..."       │
│    Doc 2: "Data scientist..."           │
│    Doc 3: "ML engineer..."              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. LLM GENERATION (Optional)            │
│    Context: [Doc 1, Doc 2, Doc 3]       │
│    Question: "Who are the AI experts?"  │
│    → Generate natural language answer   │
└─────────────────────────────────────────┘
    ↓
RESPONSE: "The dataset contains computer 
scientists and data scientists specializing 
in AI and machine learning..."
```

---

## 🔧 Files Created During Execution

```
project/
├── data/                          # Created in Step 2
│   ├── persona_0.txt
│   ├── persona_1.txt
│   └── ... (100 files)
│
└── lancedb_data/                  # Created in Step 5
    └── personas_rag.lance/        # Vector database
        ├── data/                  # Vector embeddings
        ├── _versions/             # Version control
        └── _indices/              # Search indices
```

---

## ⏱️ Execution Timeline

```
Time    Step                        Duration
─────────────────────────────────────────────
0:00    Import libraries            2 sec
0:02    Load HuggingFace dataset    3 sec
0:05    Setup LanceDB               1 sec
0:06    Download embedding model    10 sec
0:16    Generate embeddings         15 sec
0:31    Store in LanceDB            2 sec
0:33    Test vector search          1 sec
0:34    Show usage examples         <1 sec
─────────────────────────────────────────────
Total                               ~35 sec
```

---

## 💡 Key Concepts

### **RAG = Retrieval Augmented Generation**

1. **Retrieval:** Find relevant documents (vector search)
2. **Augmented:** Add context to the question
3. **Generation:** LLM generates answer using context

**Why RAG?**
- LLMs have limited context windows
- Can't memorize all your data
- Solution: Retrieve relevant parts + generate answer

---

## 🎯 Summary

**The code executes in this order:**

1. ✅ **Import** all libraries
2. ✅ **Load** 100 personas from HuggingFace
3. ✅ **Connect** to LanceDB database
4. ✅ **Create** embeddings for all documents
5. ✅ **Store** vectors in LanceDB
6. ✅ **Test** vector search with 4 queries
7. ✅ **Display** usage examples

**Result:** A fully functional RAG system ready for queries!

---

## 🚀 What You Can Do Next

1. **Query the system:**
   ```bash
   python quick_test.py "your question"
   ```

2. **Use the UI:**
   ```bash
   streamlit run app.py
   # or
   python app_gradio.py
   ```

3. **Explore the database:**
   ```python
   explore_lancedb_table(db, table_name)
   ```

4. **Add your own data:**
   - Replace `prepare_data()` function
   - Load your documents instead of personas
   - Run the same pipeline!

---

This RAG system is now ready to answer questions about the 100 personas in your dataset! 🎉



