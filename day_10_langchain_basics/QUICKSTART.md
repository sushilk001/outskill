# Quick Start Guide - Day 10 LangChain Application

Get up and running with the LangChain application in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation Steps

### 1. Navigate to the project directory

```bash
cd day10
```

### 2. Create a virtual environment (recommended)

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- LangChain and related libraries
- OpenAI SDK
- Vector stores (FAISS, ChromaDB, LanceDB)
- Web UI frameworks (Streamlit, Gradio)

### 4. Set up your API key

Create a `.env` file in the `day10` directory:

```bash
# Option 1: Create manually
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Option 2: Use a text editor
nano .env  # or vim, code, etc.
```

Add your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important:** Replace `sk-your-actual-api-key-here` with your actual OpenAI API key!

## Running the Applications

### Option 1: Basic Chain Examples (Recommended for beginners)

Start with simple examples to understand LangChain basics:

```bash
python basic_chain.py
```

This will demonstrate:
- Simple LLM calls
- Prompt templates
- Chains
- Sequential processing

### Option 2: Conversation App

Try a chatbot with memory:

```bash
python conversation_app.py
```

Features:
- Different memory types
- Conversation history
- Interactive chat mode

### Option 3: RAG Application

Build a question-answering system with documents:

```bash
python rag_app.py
```

This will:
- Create sample documents
- Build a vector database
- Enable Q&A on the documents

### Option 4: Agent with Tools

Explore autonomous agents:

```bash
python agent_app.py
```

The agent can:
- Perform calculations
- Work with files
- Process text
- Combine multiple tools

### Option 5: Web UI (Most User-Friendly)

Launch the interactive Streamlit interface:

```bash
streamlit run streamlit_app.py
```

This opens a web browser with:
- Chat interface
- Prompt playground
- Text analysis tools
- No coding required!

## Example Workflow

Here's a suggested learning path:

1. **Start with basics** (10 minutes):
   ```bash
   python basic_chain.py
   ```
   Read through the code to understand fundamental concepts.

2. **Try conversations** (10 minutes):
   ```bash
   python conversation_app.py
   ```
   Experiment with different memory types.

3. **Explore RAG** (15 minutes):
   ```bash
   python rag_app.py
   ```
   See how to add external knowledge to LLMs.

4. **Play with agents** (15 minutes):
   ```bash
   python agent_app.py
   ```
   Watch autonomous task execution.

5. **Launch the web UI** (open-ended):
   ```bash
   streamlit run streamlit_app.py
   ```
   Experiment with everything in a visual interface!

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you're in the virtual environment
pip install -r requirements.txt --upgrade
```

### "API key not found" errors

1. Check your `.env` file exists in the `day10` directory
2. Verify the API key is correct
3. Make sure there are no extra spaces or quotes

### "Rate limit" errors

- You've exceeded your OpenAI API quota
- Check your usage at https://platform.openai.com/usage
- Add credits to your account if needed

### Slow responses

- First run downloads embedding models (~100MB)
- Subsequent runs will be faster
- Consider using a faster model like `gpt-3.5-turbo`

## Next Steps

Once you're comfortable with the basics:

1. **Explore examples**:
   ```bash
   python examples/custom_chains.py
   python examples/prompt_templates.py
   ```

2. **Modify the code**:
   - Change prompts in the applications
   - Add your own documents to RAG
   - Create custom tools for the agent

3. **Build something new**:
   - Create a specialized chatbot
   - Build a document Q&A system with your data
   - Combine multiple features into a custom app

## Learning Resources

- **LangChain Docs**: https://python.langchain.com/
- **OpenAI API Docs**: https://platform.openai.com/docs
- **This Project's README**: See `README.md` for detailed documentation

## Need Help?

- Check the main `README.md` for detailed documentation
- Review code comments in each file
- Try the examples in the `examples/` directory
- Experiment with the web UI for instant feedback

Happy coding! 🚀

