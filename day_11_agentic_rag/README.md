# Agentic RAG System

Intelligent document Q&A with automatic routing between PDF documents and web search using LangChain.

## 🎯 How It Works

```
User Question → Retrieve from PDF → Check Relevance → Route Decision
                                          ↓
                               ┌──────────┴──────────┐
                               ↓                     ↓
                          [RELEVANT]            [NOT RELEVANT]
                               ↓                     ↓
                        Answer from PDF         Web Search
                               └──────────┬──────────┘
                                         ↓
                                    Return Answer
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
cd day_11
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```env
OPENAI_API_KEY=your_key_here
```

Get your key: https://platform.openai.com/api-keys

### 3. Add PDFs (Optional)
```bash
cp your-document.pdf data/pdfs/
```

### 4. Run
```bash
streamlit run streamlit_app.py
```

Open: http://localhost:8501

## 📁 Files

```
day_11/
├── config.py              # Configuration settings
├── agentic_rag_app.py    # Core RAG system
├── streamlit_app.py      # Web interface
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
├── .gitignore           # Git ignore rules
└── data/pdfs/           # Your PDF files
```

## 🎮 Usage

### In the UI:

1. **Enter API key** in sidebar
2. **Add PDFs** to `data/pdfs/` folder
3. **Click "Load PDFs"**
4. **Ask questions!**

### Example Questions:

**For PDFs:**
```
- What is this document about?
- Summarize the key points
- What are the main findings?
```

**General (triggers web search):**
```
- What is machine learning?
- Explain LangChain
- Current weather in Paris
```

## ⚙️ Configuration

Edit `.env` to customize:

```env
# Model (gpt-3.5-turbo, gpt-4)
DEFAULT_MODEL=gpt-3.5-turbo

# Temperature (0.0=focused, 1.0=creative)
TEMPERATURE=0.7

# Chunks to retrieve
TOP_K_RESULTS=3

# Document chunk size
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🔧 Alternative: OpenRouter

Use multiple models via OpenRouter:

```env
USE_OPENROUTER=true
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o
```

Get key: https://openrouter.ai/keys

## 🌐 Enable Real Web Search

Currently web search is simulated. To enable:

### Using Tavily (Recommended)
```bash
pip install tavily-python
```

Add to `.env`:
```env
TAVILY_API_KEY=your_key_here
```

Update `agentic_rag_app.py`:
```python
from tavily import TavilyClient

def web_search_simulation(query: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query, max_results=5)
    return format_results(response)
```

## 🐛 Troubleshooting

### "API Key not set"
- Check `.env` file exists in day_11 folder
- Verify key is correct (starts with `sk-`)

### "No PDFs found"
- Check files are in `data/pdfs/` directory
- Ensure files have `.pdf` extension
- Click "Load PDFs" button in UI

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

### Port already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 🧠 Key Features

✅ **Smart Routing**: Agent decides PDF vs Web automatically
✅ **Semantic Search**: FAISS vector store with HuggingFace embeddings
✅ **Multi-model**: OpenAI, OpenRouter support
✅ **Source Attribution**: Shows answer source (PDF/Web)
✅ **Chat History**: Maintains conversation
✅ **Real-time Config**: Adjust settings in UI

## 📚 Tech Stack

- **LangChain** - LLM framework
- **OpenAI** - Language models
- **FAISS** - Vector database
- **HuggingFace** - Embeddings (all-MiniLM-L6-v2)
- **Streamlit** - Web UI
- **PyPDF** - PDF processing

## 🎓 How It Works

### 1. Document Processing
```python
# Load PDFs → Split into chunks → Generate embeddings → Store in FAISS
pdfs → chunks (1000 chars) → vectors (384-dim) → FAISS index
```

### 2. Query Processing
```python
# Question → Retrieve context → Check relevance → Route & Generate
question → top 3 chunks → LLM evaluates → PDF or Web → answer
```

### 3. Intelligent Routing
```python
# Agent-based decision
relevance = check_relevance(question, context)
if relevance["is_relevant"]:
    return generate_from_pdf()  # 📄
else:
    return generate_from_web()  # 🌐
```

## 🚀 Advanced Usage

### Python Script
```python
from agentic_rag_app import AgenticRAGSystem

# Initialize
rag = AgenticRAGSystem()
rag.load_pdfs()

# Ask question
result = rag.answer_question("What is this about?")
print(result['answer'])
print(f"Source: {result['source']}")  # 'pdf' or 'web'
```

### CLI Mode
```bash
python agentic_rag_app.py
```

## 📊 Performance

| Operation | Time |
|-----------|------|
| PDF Loading | 2-10s (one-time) |
| Query | 3-8s |
| Retrieval | <0.5s |
| Relevance Check | 1-2s |
| Answer Generation | 2-5s |

## 🔒 Privacy

- All processing is local
- Documents never leave your machine
- Only queries/responses sent to LLM provider
- Vector stores cached locally

## 📝 Example `.env`

```env
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
DEFAULT_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
TOP_K_RESULTS=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Alternative: OpenRouter
# USE_OPENROUTER=true
# OPENROUTER_API_KEY=your-key
# OPENROUTER_MODEL=openai/gpt-4o

# Future: Real web search
# TAVILY_API_KEY=your-tavily-key
```

## 🎉 That's It!

Lightweight, powerful, and easy to use. Start asking questions about your documents!

---

**Questions?** Check the code comments in `agentic_rag_app.py` and `streamlit_app.py`

**Part of Outskill Day 11** | Educational Project
