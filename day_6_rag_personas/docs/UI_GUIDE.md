# 🎨 Interactive UI Guide

## Two Beautiful UI Options Available!

### Option 1: Streamlit (Recommended) 🌟

**Features:**
- Chat-like interface
- Real-time conversation
- Beautiful, modern design
- Sidebar with settings
- Sample query buttons
- Chat history

**Launch:**
```bash
streamlit run app.py
```

**Access:** Opens automatically in your browser at `http://localhost:8501`

---

### Option 2: Gradio 🎯

**Features:**
- Clean, simple interface
- Side-by-side results
- Easy to share
- Sample query buttons
- Multiple search modes

**Launch:**
```bash
python app_gradio.py
```

**Access:** Opens automatically in your browser at `http://localhost:7860`

---

## 🚀 Quick Start

### 1. Launch Streamlit UI (Recommended)
```bash
streamlit run app.py
```

### 2. Launch Gradio UI
```bash
python app_gradio.py
```

---

## 🎨 Features Comparison

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| Chat Interface | ✅ Yes | ❌ No |
| Chat History | ✅ Yes | ❌ No |
| Modern Design | ✅✅ Excellent | ✅ Good |
| Easy Sharing | ❌ No | ✅ Yes |
| Vector Search | ✅ Yes | ✅ Yes |
| RAG with LLM | ✅ Yes | ✅ Yes |
| Sample Queries | ✅ Yes | ✅ Yes |
| Adjustable Settings | ✅ Yes | ✅ Yes |

---

## 📸 What to Expect

### Streamlit Interface:
- **Left Panel:** Settings and sample queries
- **Main Area:** Chat interface with your questions and AI responses
- **Real-time:** Type and get instant responses
- **History:** All your queries saved in the session

### Gradio Interface:
- **Top:** Query input and search mode selector
- **Middle:** Sample query buttons
- **Bottom:** Split view with AI response and supporting documents
- **Simple:** Clean and straightforward

---

## 💡 Tips

1. **First Query is Slower**: Model loading takes a few seconds
2. **Subsequent Queries**: Much faster after initialization
3. **Vector Search**: Fastest option, no LLM generation
4. **RAG with LLM**: Best quality responses
5. **Both Mode**: See both vector results and AI response

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# For Streamlit (change port)
streamlit run app.py --server.port 8502

# For Gradio (change port in app_gradio.py)
# Edit line: server_port=7861
```

### System Not Ready
Make sure:
1. Run `python RAG_Implementation.py` first
2. Ollama is running
3. `lancedb_data/` directory exists

### Slow Performance
- Use "Vector Search Only" mode for faster results
- Reduce number of documents in settings
- First query is always slower (model loading)

---

## 🎯 Sample Queries to Try

**Technology:**
- "Who are the AI and machine learning experts?"
- "Find software developers and engineers"
- "Tell me about data scientists"

**Education:**
- "Who are the teachers and educators?"
- "Find professors and instructors"
- "Show me people in education"

**Environment:**
- "Who focuses on climate change?"
- "Find environmental scientists"
- "Tell me about sustainability experts"

**Healthcare:**
- "Who works in healthcare?"
- "Find medical professionals"
- "Tell me about doctors and nurses"

**Creative:**
- "Who are the artists and designers?"
- "Find people in creative fields"
- "Show me writers and content creators"

---

## 🌐 Sharing Your UI

### Streamlit Cloud (Free)
1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Deploy your app
4. Get a public URL

### Gradio Share
```python
# In app_gradio.py, change:
demo.launch(share=True)  # Creates a public link
```

---

## 📝 Customization

### Change Theme (Streamlit)
Add to `app.py`:
```python
st.set_page_config(
    page_title="My Custom Title",
    page_icon="🔥",
    theme={
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF"
    }
)
```

### Change Colors (Gradio)
```python
demo = gr.Blocks(theme=gr.themes.Base())  # or Soft(), Glass(), etc.
```

---

## 🎉 Enjoy Your Interactive RAG System!

Choose the interface you prefer and start exploring your data!



