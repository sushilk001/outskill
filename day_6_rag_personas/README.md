# RAG Implementation Testing Guide

## 🎯 Three Ways to Test Your RAG System

### ✅ Prerequisites
1. Run the main implementation first:
   ```bash
   python RAG_Implementation.py
   ```
2. Make sure Ollama is running with gemma3:1b model
3. Check that the `lancedb_data/` directory was created

---

## Method 1: Quick Test (Command Line)

**Best for**: Single queries, scripting, quick tests

```bash
# Basic usage
python quick_test.py "your question here"

# Examples
python quick_test.py "Who are the AI experts?"
python quick_test.py "Find teachers in the dataset"
python quick_test.py "Tell me about environmental scientists"
```

**Output**: Shows both the RAG response and top 3 relevant documents

---

## Method 2: Interactive Mode

**Best for**: Multiple queries, exploration, experimentation

```bash
python test_rag.py
```

### Features:
- **Interactive prompt**: Type queries and get instant responses
- **Multiple modes**:
  - Default: RAG with LLM response
  - `vector <query>`: Vector search only (faster)
  - `both <query>`: Show both vector search and RAG response
  - `samples`: Run predefined test queries
  - `quit` or `exit`: Exit the program

### Example Session:
```
💬 Your query: Who are the technology experts?
🤖 RAG with LLM Response:
[Response here]

💬 Your query: vector climate change experts
🔍 Vector Search Results (Top 3):
[Results here]

💬 Your query: both teachers in dataset
🔍 Vector Search Results + 🤖 RAG Response
[Both results here]
```

---

## Method 3: Modify the Main Script

Edit `RAG_Implementation.py` and add your custom queries:

```python
# At the end of main() function, add:
async def main():
    # ... existing code ...
    
    # Add your custom queries
    custom_queries = [
        "Your question 1",
        "Your question 2",
        "Your question 3"
    ]
    
    for query in custom_queries:
        print(f"\nQuery: {query}")
        response = query_rag(query_engine, query)
        print(f"Response: {response}")
```

---

## 📝 Sample Test Queries

Try these queries to explore your dataset:

### Technology & AI
- "Who are the artificial intelligence experts?"
- "Find people working with machine learning"
- "Tell me about computer scientists in the dataset"

### Education
- "Who are the teachers or educators?"
- "Find professors and instructors"
- "Show me people working in education"

### Environment
- "Who focuses on climate change?"
- "Find environmental scientists"
- "Tell me about sustainability experts"

### Creative Fields
- "Who are the artists in the dataset?"
- "Find people working in creative industries"
- "Show me designers and creative professionals"

### Healthcare
- "Find medical professionals"
- "Who are the healthcare workers?"
- "Tell me about doctors and nurses"

### Data & Research
- "Who are the data scientists?"
- "Find researchers and analysts"
- "Show me people working with statistics"

---

## 🔧 Troubleshooting

### Error: "LanceDB data not found"
**Solution**: Run `python RAG_Implementation.py` first

### Error: Connection to Ollama failed
**Solution**: Make sure Ollama is running
```bash
# Check if Ollama is running
ollama list

# If not, the service should auto-start, or try:
# On Mac/Linux: just run 'ollama serve' in another terminal
```

### Error: Model not found
**Solution**: Pull the gemma3:1b model
```bash
ollama pull gemma3:1b
```

### Slow responses
- First query is slower (model loading)
- Subsequent queries are faster
- Use `vector` mode for fastest results (no LLM)

---

## 🎨 Customization

### Change the model
Edit the scripts and replace `gemma3:1b` with another model:
```python
llm = Ollama(
    model="llama3.2:1b",  # or any other model
    base_url="http://localhost:11434",
    request_timeout=60.0
)
```

### Adjust number of results
Change `top_k` parameter:
```python
test_vector_search_only(query, embed_model, top_k=5)  # Show 5 results
```

### Use different embedding models
```python
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## 📊 Understanding the Results

### Similarity Scores
- **Lower is better** (0.0 = perfect match)
- < 0.7: Highly relevant
- 0.7-1.0: Moderately relevant  
- \> 1.0: Less relevant

### Response Types
- **Vector Search**: Fast retrieval, no generation
- **RAG with LLM**: Contextual, generated response using retrieved documents

---

## 🚀 Next Steps

1. Test with your own datasets
2. Experiment with different models
3. Try different embedding models
4. Adjust chunk sizes and overlap
5. Add metadata filtering
6. Implement custom response modes

Happy testing! 🎉






