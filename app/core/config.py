from pydantic import BaseSettings
from functools import lru_cache
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings management using Pydantic.
    
    This class manages all configuration settings for the application,
    supporting both development and production environments.
    """
    
    # Application Settings
    APP_NAME: str = "VectorHub"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Vector Database Settings
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 8080
    VECTOR_DB_BATCH_SIZE: int = 100
    VECTOR_DB_TIMEOUT_RETRIES: int = 3
    VECTOR_DB_TIMEOUT_SECONDS: int = 300
    
    # Embedding Model Settings
    DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_MAX_LENGTH: int = 512
    USE_CUDA: Optional[bool] = None  # None means auto-detect
    
    # Cache Settings
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 10000
    
    # Security Settings
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: Optional[str] = None
    CORS_ORIGINS: list = ["*"]
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        """Pydantic configuration class."""
        env_file = ".env"
        case_sensitive = True

    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration dictionary.
        
        Returns:
            Dictionary containing vector database configuration settings.
        """
        return {
            "host": self.VECTOR_DB_HOST,
            "port": self.VECTOR_DB_PORT,
            "batch_size": self.VECTOR_DB_BATCH_SIZE,
            "timeout_config": {
                "timeout_retries": self.VECTOR_DB_TIMEOUT_RETRIES,
                "timeout_seconds": self.VECTOR_DB_TIMEOUT_SECONDS
            }
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration dictionary.
        
        Returns:
            Dictionary containing embedding model configuration settings.
        """
        return {
            "model_name": self.DEFAULT_MODEL_NAME,
            "max_length": self.MODEL_MAX_LENGTH,
            "device": "cuda" if self.USE_CUDA is True or 
                    (self.USE_CUDA is None and torch.cuda.is_available()) 
                    else "cpu"
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration dictionary.
        
        Returns:
            Dictionary containing cache configuration settings.
        """
        return {
            "enabled": self.CACHE_ENABLED,
            "ttl": self.CACHE_TTL,
            "max_size": self.CACHE_MAX_SIZE
        }

@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance.
    
    Returns:
        Cached Settings instance.
    """
    return Settings()

# Example .env file template
def create_env_template():
    """Create a template .env file with default values."""
    template = """# Application Settings
APP_NAME=VectorHub
APP_VERSION=1.0.0
DEBUG=False

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Vector Database Settings
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=8080
VECTOR_DB_BATCH_SIZE=100
VECTOR_DB_TIMEOUT_RETRIES=3
VECTOR_DB_TIMEOUT_SECONDS=300

# Embedding Model Settings
DEFAULT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
MODEL_MAX_LENGTH=512
USE_CUDA=

# Cache Settings
CACHE_ENABLED=True
CACHE_TTL=3600
CACHE_MAX_SIZE=10000

# Security Settings
API_KEY=
CORS_ORIGINS=*

# Logging Settings
LOG_LEVEL=INFO
"""
    
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        with open(env_path, "