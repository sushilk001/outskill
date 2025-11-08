"""
Agentic RAG System - Intelligent Document Q&A with Web Search Fallback
Robust version with comprehensive error handling and validation
"""

import os
import logging
from typing import List, Dict, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenticRAGSystem:
    """Agentic RAG system with intelligent routing and robust error handling"""
    
    def __init__(self):
        self.config = Config
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        
        # Initialize components with error handling
        try:
            self._initialize_embeddings()
            self._initialize_llm()
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}", exc_info=True)
            # Continue with None values - will be initialized later
    
    def _initialize_llm(self) -> bool:
        """Initialize the language model with robust error handling"""
        try:
            if not self.config.OPENAI_API_KEY and not self.config.OPENROUTER_API_KEY:
                logger.warning("No API key configured")
                self.llm = None
                return False
            
            if self.config.USE_OPENROUTER and self.config.OPENROUTER_API_KEY:
                api_key = self.config.OPENROUTER_API_KEY.strip()
                if not api_key or len(api_key) < 10:
                    logger.warning(f"OpenRouter API key is invalid (length: {len(api_key)})")
                    self.llm = None
                    return False
                
                # Validate key format (should start with sk-)
                if not api_key.startswith('sk-'):
                    logger.warning("OpenRouter API key should start with 'sk-'")
                    
                self.llm = ChatOpenAI(
                    model=self.config.OPENROUTER_MODEL,
                    openai_api_key=api_key,
                    openai_api_base=self.config.OPENROUTER_BASE_URL,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    timeout=30.0,
                )
                logger.info(f"LLM initialized: {self.config.OPENROUTER_MODEL}")
                return True
                
            elif self.config.OPENAI_API_KEY:
                api_key = self.config.OPENAI_API_KEY.strip()
                if not api_key or len(api_key) < 10:
                    logger.warning(f"OpenAI API key is invalid (length: {len(api_key)})")
                    self.llm = None
                    return False
                
                # Validate key format (should start with sk-)
                if not api_key.startswith('sk-'):
                    logger.warning("OpenAI API key should start with 'sk-'. Please check your API key.")
                    self.llm = None
                    return False
                    
                self.llm = ChatOpenAI(
                    model=self.config.DEFAULT_MODEL,
                    api_key=api_key,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    timeout=30.0,
                )
                logger.info(f"LLM initialized: {self.config.DEFAULT_MODEL}")
                return True
            else:
                logger.warning("No valid API key found")
                self.llm = None
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "auth" in error_msg.lower():
                logger.error(f"Authentication failed - API key is invalid: {e}")
            else:
                logger.error(f"Error initializing LLM: {e}", exc_info=True)
            self.llm = None
            return False
    
    def reinitialize_llm(self) -> bool:
        """Reinitialize LLM (useful when API key is updated)"""
        return self._initialize_llm()
    
    def _initialize_embeddings(self) -> bool:
        """Initialize embedding model with error handling"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"Embeddings initialized: {self.config.EMBEDDING_MODEL}")
            return True
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}", exc_info=True)
            self.embeddings = None
            return False
    
    def load_pdfs(self, pdf_directory: Optional[str] = None) -> int:
        """Load and process PDF documents with comprehensive error handling"""
        pdf_dir = pdf_directory or self.config.PDF_DIR
        
        try:
            # Validate directory
            if not pdf_dir or not isinstance(pdf_dir, str):
                raise ValueError(f"Invalid PDF directory: {pdf_dir}")
            
            if not os.path.exists(pdf_dir):
                raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
            
            if not os.path.isdir(pdf_dir):
                raise ValueError(f"Path is not a directory: {pdf_dir}")
            
            # Check for PDF files
            try:
                pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
            except PermissionError:
                raise PermissionError(f"Permission denied accessing directory: {pdf_dir}")
            
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")
            
            logger.info(f"Found {len(pdf_files)} PDF file(s)")
            
            # Load PDFs with error handling
            try:
                loader = DirectoryLoader(
                    pdf_dir,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=False  # Disable progress for cleaner logs
                )
                documents = loader.load()
            except Exception as e:
                raise RuntimeError(f"Error loading PDFs: {e}")
            
            if not documents:
                raise ValueError("No documents loaded from PDFs")
            
            logger.info(f"Loaded {len(documents)} pages from PDFs")
            
            # Validate embeddings
            if not self.embeddings:
                raise RuntimeError("Embeddings not initialized")
            
            # Split documents into chunks
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max(100, self.config.CHUNK_SIZE),  # Ensure minimum size
                    chunk_overlap=max(0, min(self.config.CHUNK_OVERLAP, self.config.CHUNK_SIZE // 2)),
                    length_function=len,
                )
                chunks = text_splitter.split_documents(documents)
            except Exception as e:
                raise RuntimeError(f"Error splitting documents: {e}")
            
            if not chunks:
                raise ValueError("No chunks created from documents")
            
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Create vector store
            try:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            except Exception as e:
                raise RuntimeError(f"Error creating vector store: {e}")
            
            logger.info("Vector store created successfully")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}", exc_info=True)
            raise
    
    def retrieve_from_pdf(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant context from PDF documents with error handling"""
        # Input validation
        if not query or not isinstance(query, str):
            return {
                "success": False,
                "error": "Invalid query: must be a non-empty string",
                "context": "",
                "documents": [],
                "num_docs": 0
            }
        
        query = query.strip()
        if not query:
            return {
                "success": False,
                "error": "Query cannot be empty",
                "context": "",
                "documents": [],
                "num_docs": 0
            }
        
        if not self.vector_store:
            return {
                "success": False,
                "error": "Vector store not initialized. Please load PDFs first.",
                "context": "",
                "documents": [],
                "num_docs": 0
            }
        
        try:
            # Retrieve relevant documents
            k = max(1, min(self.config.TOP_K_RESULTS, 10))  # Clamp between 1-10
            docs = self.vector_store.similarity_search(query, k=k)
            
            if not docs:
                return {
                    "success": True,
                    "context": "",
                    "documents": [],
                    "num_docs": 0,
                    "warning": "No relevant documents found"
                }
            
            # Safely combine context
            try:
                context_parts = []
                for doc in docs:
                    if hasattr(doc, 'page_content'):
                        content = str(doc.page_content).strip()
                        if content:
                            context_parts.append(content)
                
                context = "\n\n".join(context_parts)
            except Exception as e:
                logger.warning(f"Error combining context: {e}")
                context = ""
            
            return {
                "success": True,
                "context": context,
                "documents": docs,
                "num_docs": len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving from PDF: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Retrieval error: {str(e)}",
                "context": "",
                "documents": [],
                "num_docs": 0
            }
    
    def check_relevance(self, query: str, context: str) -> Dict[str, Any]:
        """Check if retrieved context is relevant to the query"""
        # Input validation
        if not query or not isinstance(query, str):
            return {
                "is_relevant": False,
                "decision": "NOT_RELEVANT (invalid query)",
                "reasoning": "Invalid query provided"
            }
        
        if not context or not isinstance(context, str):
            return {
                "is_relevant": False,
                "decision": "NOT_RELEVANT (no context)",
                "reasoning": "No context provided"
            }
        
        if not self.llm:
            return {
                "is_relevant": False,
                "decision": "NOT_RELEVANT (LLM not initialized)",
                "reasoning": "LLM not initialized. Please configure API key."
            }
        
        # Limit context length to avoid token limits
        max_context_length = 2000
        truncated_context = context[:max_context_length]
        if len(context) > max_context_length:
            truncated_context += "..."
        
        relevance_prompt = f"""You are a relevance checker. Determine if the context can answer the question.

Question: {query.strip()}

Context: {truncated_context}

Respond with ONLY one word: "RELEVANT" or "NOT_RELEVANT"."""
        
        try:
            response = self.llm.invoke(relevance_prompt)
            
            # Safely extract content
            if hasattr(response, 'content'):
                decision_text = str(response.content).strip().upper()
            else:
                decision_text = str(response).strip().upper()
            
            # Parse decision
            is_relevant = "RELEVANT" in decision_text and "NOT_RELEVANT" not in decision_text
            
            return {
                "is_relevant": is_relevant,
                "decision": decision_text[:100],  # Limit length
                "reasoning": decision_text
            }
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}", exc_info=True)
            # Default to relevant to allow fallback
            try:
                error_msg = str(e)
                error_msg = error_msg.encode('ascii', errors='replace').decode('ascii')
            except Exception:
                error_msg = "Error occurred"
            return {
                "is_relevant": True,
                "decision": "RELEVANT (error fallback)",
                "reasoning": f"Error occurred: {error_msg[:100]}"
            }
    
    def generate_answer_from_pdf(self, query: str, context: str) -> str:
        """Generate answer using PDF context with error handling"""
        # Input validation
        if not query or not isinstance(query, str) or not query.strip():
            return "Error: Invalid query provided."
        
        if not context or not isinstance(context, str):
            return "Error: No context provided."
        
        if not self.llm:
            return "Error: LLM not initialized. Please enter a valid OpenAI API key in the sidebar (must start with 'sk-')."
        
        # Limit context length
        max_context = 3000
        safe_context = context[:max_context] if len(context) > max_context else context
        
        answer_prompt = f"""Answer the question using ONLY the provided context.

Context: {safe_context}

Question: {query.strip()}

Answer concisely based on the context. If the context doesn't fully answer, say what you know and what's missing."""
        
        try:
            response = self.llm.invoke(answer_prompt)
            
            # Safely extract content
            if hasattr(response, 'content'):
                answer = str(response.content)
            else:
                answer = str(response)
            
            # Validate answer
            if not answer or len(answer.strip()) < 10:
                return "Error: Received empty or invalid response from LLM."
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer from PDF: {e}", exc_info=True)
            # Safely handle error message with Unicode
            try:
                error_msg = str(e)
                # Remove or replace problematic Unicode characters
                error_msg = error_msg.encode('ascii', errors='replace').decode('ascii')
            except Exception:
                error_msg = "An error occurred while generating the answer"
            return f"Error generating answer: {error_msg[:200]}"
    
    @staticmethod
    def web_search_simulation(query: str) -> str:
        """Simulate web search (placeholder)"""
        if not query or not isinstance(query, str):
            return "[No search query provided]"
        
        return f"""Simulated web search results for: "{query[:100]}"

Note: This is a placeholder. To enable real web search:
1. Install: pip install tavily-python
2. Get API key from https://tavily.com
3. Update the web_search_simulation method"""
    
    def generate_answer_from_web(self, query: str) -> str:
        """Generate answer using web search results"""
        # Input validation
        if not query or not isinstance(query, str) or not query.strip():
            return "Error: Invalid query provided."
        
        if not self.llm:
            return "Error: LLM not initialized. Please enter a valid OpenAI API key in the sidebar (must start with 'sk-')."
        
        # Perform web search
        try:
            web_results = self.web_search_simulation(query)
        except Exception as e:
            logger.warning(f"Error in web search simulation: {e}")
            web_results = "[Web search unavailable]"
        
        answer_prompt = f"""Answer the question based on web search results.

Web Results: {web_results[:1000]}

Question: {query.strip()}

Provide a helpful answer. If results are simulated, acknowledge this."""
        
        try:
            response = self.llm.invoke(answer_prompt)
            
            # Safely extract content
            if hasattr(response, 'content'):
                answer = str(response.content)
            else:
                answer = str(response)
            
            if not answer or len(answer.strip()) < 10:
                return "Error: Received empty response from LLM."
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer from web: {e}", exc_info=True)
            # Safely handle error message with Unicode
            try:
                error_msg = str(e)
                # Remove or replace problematic Unicode characters
                error_msg = error_msg.encode('ascii', errors='replace').decode('ascii')
            except Exception:
                error_msg = "An error occurred while generating the answer"
            return f"Error generating answer: {error_msg[:200]}"
    
    def answer_question(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Main method: Answer question using agentic RAG flow
        Returns a dictionary with success status, answer, source, and metadata
        """
        # Input validation
        if not query:
            return {
                "success": False,
                "answer": "Error: Query cannot be empty.",
                "source": "error",
                "error": "Empty query"
            }
        
        if not isinstance(query, str):
            return {
                "success": False,
                "answer": "Error: Query must be a string.",
                "source": "error",
                "error": "Invalid query type"
            }
        
        query = query.strip()
        if not query:
            return {
                "success": False,
                "answer": "Error: Query cannot be empty.",
                "source": "error",
                "error": "Empty query after strip"
            }
        
        try:
            # Step 1: Retrieve from PDF
            retrieval_result = self.retrieve_from_pdf(query)
            
            if not retrieval_result.get("success"):
                # If retrieval fails, try web search
                logger.warning(f"PDF retrieval failed: {retrieval_result.get('error')}")
                try:
                    answer = self.generate_answer_from_web(query)
                    return {
                        "success": True,
                        "answer": answer,
                        "source": "web",
                        "retrieval": retrieval_result
                    }
                except Exception as e:
                    logger.error(f"Error in web fallback: {e}")
                    return {
                        "success": False,
                        "answer": f"Error: {retrieval_result.get('error', 'Unknown error')}",
                        "source": "error",
                        "error": str(e)
                    }
            
            context = retrieval_result.get("context", "")
            
            # Step 2: Check relevance (only if we have context)
            if context:
                try:
                    relevance = self.check_relevance(query, context)
                except Exception as e:
                    logger.error(f"Error checking relevance: {e}")
                    relevance = {"is_relevant": True, "decision": "RELEVANT (error fallback)"}
            else:
                relevance = {"is_relevant": False, "decision": "NOT_RELEVANT (no context)"}
            
            # Step 3: Route based on relevance
            try:
                if relevance.get("is_relevant", False):
                    answer = self.generate_answer_from_pdf(query, context)
                    source = "pdf"
                else:
                    answer = self.generate_answer_from_web(query)
                    source = "web"
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                # Safely handle error message with Unicode
                try:
                    error_msg = str(e)
                    error_msg = error_msg.encode('ascii', errors='replace').decode('ascii')
                except Exception:
                    error_msg = "An error occurred while generating the answer"
                return {
                    "success": False,
                    "answer": f"Error generating answer: {error_msg[:200]}",
                    "source": "error",
                    "error": error_msg
                }
            
            # Validate answer
            if not answer or len(answer.strip()) < 5:
                return {
                    "success": False,
                    "answer": "Error: Received empty or invalid answer.",
                    "source": "error",
                    "error": "Empty answer"
                }
            
            # Step 4: Return result
            return {
                "success": True,
                "answer": answer,
                "source": source,
                "relevance": relevance,
                "retrieval": retrieval_result
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            # Safely handle error message with Unicode
            try:
                error_msg = str(e)
                error_msg = error_msg.encode('ascii', errors='replace').decode('ascii')
            except Exception:
                error_msg = "An error occurred while processing the question"
            return {
                "success": False,
                "answer": f"Error processing question: {error_msg[:200]}",
                "source": "error",
                "error": error_msg
            }


def main():
    """Main function for CLI usage"""
    try:
        logger.info("Starting Agentic RAG System")
        
        if not Config.validate():
            logger.error("Configuration validation failed")
            return
        
        rag_system = AgenticRAGSystem()
        
        # Try to load PDFs
        try:
            num_chunks = rag_system.load_pdfs()
            logger.info(f"System ready with {num_chunks} chunks")
        except Exception as e:
            logger.warning(f"Could not load PDFs: {e}")
        
        # Interactive mode
        print("\n" + "="*70)
        print("Agentic RAG System - Interactive Mode")
        print("Type 'quit' to exit")
        print("="*70)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                result = rag_system.answer_question(question, verbose=False)
                
                if result.get("success"):
                    print(f"\nAnswer ({result.get('source', 'unknown')}):")
                    print(result.get("answer", "No answer"))
                else:
                    print(f"\nError: {result.get('answer', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}")
                print(f"Error: {e}")
        
        logger.info("Exiting")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
