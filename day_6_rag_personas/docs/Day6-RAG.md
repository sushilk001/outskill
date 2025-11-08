## Summary of Today's Session (Day 6 - Engineering Accelerator)

### Session Overview

- **Topic**: Retrieval Augmented Generation (RAG)
- **Instructor**: Dileep (Head of AI initiatives at Outskill)
- **Format**: 7-hour Saturday session with two parts and break
- **Next Session**: Sunday 10am-2pm IST with breakout rooms
### Key RAG Concepts Covered

**Core Understanding**
- RAG is like a "super smart friend" with access to a library who must cite sources[#1][#2]
- Purpose: Reduce LLM hallucinations by providing external knowledge
- Process: Retrieve relevant information + augment prompts (not retraining)
**Technical Workflow**
1. **Data Preparation (Offline)**
    - Source collection (PDFs, videos, audio, databases)
    - Information extraction (OCR, audio-to-text)
    - Chunking (512-1024 tokens with 100-200 overlap)
    - Embedding/indexing using embedding models
    - Storage in vector databases
2. **Retrieval Process (Online)**
    - Query embedding using same model
    - Vector similarity search
    - Context retrieval and text conversion
    - Prompt augmentation with retrieved context
    - LLM generation with synthesized answer
### Tools and Services Mentioned

**Vector Databases**
- LanceDB (primary focus for session)
- Quadrant
- Pinecone
- Weaviate (VV8)
- Faiss
**Production RAG Examples**
- Notebook LM (Google)
- Perplexity AI
- Custom GPTs (OpenAI)
**Libraries and Frameworks**
- Llama Index (Meta) - end-to-end RAG orchestration
- Various embedding models
**Development Tools**
- [Dify.org](https://Dify.org) - visual RAG system builder
- Google AI Studio - can function as RAG
**Search APIs for RAG**
- Exa (EXA) - LLM-optimized search engine
- DuckDuckGo
### Key Engineering Considerations

- Trade-offs between speed vs. quality based on use case
- Token consumption and API costs
- Chunk size optimization through experimentation
- Vector database scaling and AWS/GCP billing concerns
- Customer support identified as primary commercial RAG use case
### Important Technical Notes

- RAG is augmentation, not training (model weights don't change)
- Same embedding model must be used for indexing and querying
- Debugging AI applications is probabilistic, not deterministic
- Code quality from LLMs tends to be mediocre (human oversight needed)
