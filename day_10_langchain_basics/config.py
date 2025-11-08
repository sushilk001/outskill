"""
Configuration settings for the LangChain application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # Model Settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Vector Store Settings
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss, chroma, lancedb
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Memory Settings
    MEMORY_TYPE = os.getenv("MEMORY_TYPE", "buffer")  # buffer, summary, window
    MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))
    
    # Agent Settings
    AGENT_TYPE = os.getenv("AGENT_TYPE", "openai-functions")
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
    
    # Paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_stores")
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
            print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
        
        # Create directories if they don't exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_DIR, exist_ok=True)
        
        return True
    
    @classmethod
    def get_llm_config(cls):
        """Get LLM configuration as a dictionary"""
        return {
            "model": cls.DEFAULT_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
        }


# Validate configuration on import
Config.validate()

