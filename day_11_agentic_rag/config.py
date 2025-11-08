"""
Configuration settings for the Agentic RAG application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    
    # Model Settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    
    # OpenRouter Settings (alternative to OpenAI)
    USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
    
    # Embedding Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # RAG Settings
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
    
    # Agent Settings
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    
    # Paths
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")
    VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_stores")
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY and not cls.OPENROUTER_API_KEY:
            print("⚠️  Warning: No API key set. Please set OPENAI_API_KEY or OPENROUTER_API_KEY")
            print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
            return False
        
        # Create directories if they don't exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.PDF_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_DIR, exist_ok=True)
        
        return True
    
    @classmethod
    def get_api_key(cls):
        """Get the appropriate API key"""
        if cls.USE_OPENROUTER and cls.OPENROUTER_API_KEY:
            return cls.OPENROUTER_API_KEY
        return cls.OPENAI_API_KEY
    
    @classmethod
    def get_llm_config(cls):
        """Get LLM configuration as a dictionary"""
        config = {
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
        }
        
        if cls.USE_OPENROUTER:
            config.update({
                "model": cls.OPENROUTER_MODEL,
                "openai_api_key": cls.OPENROUTER_API_KEY,
                "openai_api_base": cls.OPENROUTER_BASE_URL,
            })
        else:
            config.update({
                "model": cls.DEFAULT_MODEL,
                "api_key": cls.OPENAI_API_KEY,
            })
        
        return config


# Validate configuration on import
Config.validate()

