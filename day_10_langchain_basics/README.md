# Day 10 - LangChain Application

A comprehensive LangChain application demonstrating various capabilities including chains, prompts, memory, and agents.

## Features

- **LLM Integration**: Connect with OpenAI, Anthropic, or local models
- **Chains**: Sequential and parallel chain execution
- **Prompt Templates**: Reusable prompt engineering
- **Memory**: Conversation history and context management
- **Agents**: Autonomous task execution with tools
- **RAG**: Retrieval Augmented Generation with vector stores

## Project Structure

```
day10/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── basic_chain.py              # Simple LangChain examples
├── conversation_app.py         # Chatbot with memory
├── rag_app.py                  # RAG implementation
├── agent_app.py                # Agent with tools
├── streamlit_app.py            # Interactive web UI
└── examples/                   # Additional examples
    ├── custom_chains.py
    └── prompt_templates.py
```

## Setup

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in this directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
# Optional: Add other provider keys
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_hf_key
```

## Usage

### Basic Chain Example
```bash
python basic_chain.py
```

### Conversation App with Memory
```bash
python conversation_app.py
```

### RAG Application
```bash
python rag_app.py
```

### Agent with Tools
```bash
python agent_app.py
```

### Interactive Web UI
```bash
streamlit run streamlit_app.py
```

## Key Concepts

### 1. LLMs and Chat Models
LangChain provides abstractions for working with different language models:
- OpenAI GPT models
- Anthropic Claude
- Open-source models via HuggingFace
- Local models via Ollama

### 2. Prompts
Reusable prompt templates with variable injection:
```python
from langchain.prompts import PromptTemplate

template = "Tell me a {adjective} joke about {topic}"
prompt = PromptTemplate(template=template, input_variables=["adjective", "topic"])
```

### 3. Chains
Combine multiple components in sequence:
- LLMChain: Basic LLM + Prompt
- SequentialChain: Multiple steps in order
- RouterChain: Conditional routing

### 4. Memory
Maintain conversation context:
- ConversationBufferMemory: Keep all messages
- ConversationSummaryMemory: Summarize old messages
- ConversationBufferWindowMemory: Keep last N messages

### 5. Agents
Autonomous agents that can use tools:
- ReAct agents
- OpenAI Functions agents
- Custom tool creation

### 6. RAG (Retrieval Augmented Generation)
Enhance responses with external knowledge:
- Document loaders
- Text splitters
- Vector stores (FAISS, Chroma, LanceDB)
- Retrievers

## Examples

See the individual Python files for detailed examples and documentation.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangSmith](https://smith.langchain.com/) - Debugging and monitoring

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### API Key Issues
Verify your `.env` file is in the correct location and contains valid keys.

### Memory Issues
For large documents in RAG, consider chunking strategies and vector store optimization.

## License

Educational purposes - Part of the Outskill learning series.

