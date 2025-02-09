from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings management using Pydantic."""
    
    # Application Settings
    APP_NAME: str = "VectorSphere"
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
    USE_CUDA: Optional[bool] = False  # Changed to have a default value
    
    # Cache Settings
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 10000
    
    # Security Settings
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: Optional[str] = None
    CORS_ORIGINS: List[str] = ["*"]
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        """Pydantic configuration class."""
        env_file = ".env"
        case_sensitive = True

    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration dictionary."""
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
        """Get embedding model configuration dictionary."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        return {
            "model_name": self.DEFAULT_MODEL_NAME,
            "max_length": self.MODEL_MAX_LENGTH,
            "device": "cuda" if self.USE_CUDA and cuda_available else "cpu"
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration dictionary."""
        return {
            "enabled": self.CACHE_ENABLED,
            "ttl": self.CACHE_TTL,
            "max_size": self.CACHE_MAX_SIZE
        }

@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance."""
    return Settings()

if __name__ == "__main__":
    # Print current settings
    settings = get_settings()
    print(f"Current settings loaded: {settings.APP_NAME}")