from typing import List, Optional, Union, Dict
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from abc import ABC, abstractmethod
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text into vector embeddings."""
        pass

    @abstractmethod
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors."""
        pass

class TransformerEmbedding(BaseEmbeddingModel):
    """Custom transformer-based embedding model."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts into vector embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).tolist()

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

class EmbeddingService:
    """Service for managing multiple embedding models and caching."""
    
    def __init__(self):
        self.models: Dict[str, BaseEmbeddingModel] = {}
        self._cache = {}  # Simple in-memory cache

    def add_model(self, name: str, model: BaseEmbeddingModel):
        """Add a new embedding model to the service."""
        self.models[name] = model

    def get_embedding(
        self,
        text: Union[str, List[str]],
        model_name: str,
        use_cache: bool = True
    ) -> List[float]:
        """Get embeddings for text using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if use_cache and isinstance(text, str):
            cache_key = f"{model_name}:{text}"
            if cache_key in self._cache:
                return self._cache[cache_key]

        embeddings = self.models[model_name].encode(text)

        if use_cache and isinstance(text, str):
            self._cache[cache_key] = embeddings

        return [float(val) for val in embeddings]

embedding_service = EmbeddingService()
transformer_model = TransformerEmbedding()
embedding_service.add_model("transformer", transformer_model)

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    embedding = embedding_service.get_embedding(request.text, "transformer")
    return {"embedding": embedding}

# # Example usage
# if __name__ == "__main__":
#     # Initialize service
#     embedding_service = EmbeddingService()
    
#     # Add different embedding models
#     transformer_model = TransformerEmbedding()
#     embedding_service.add_model("transformer", transformer_model)
    
#     # Example texts
#     text1 = "This is a sample sentence."
#     text2 = "This is another example."
     
#     # Get embeddings
#     embedding1 = embedding_service.get_embedding(text1, "transformer")
#     embedding2 = embedding_service.get_embedding(text2, "transformer")
    
#     # Compute similarity
#     similarity = embedding_service.compute_similarity(text1, text2, "transformer")
#     print(f"Similarity between texts: {similarity:.4f}")