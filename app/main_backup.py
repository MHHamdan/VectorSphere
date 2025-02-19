from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import time

# Import our embedding service
from app.services.embedding import EmbeddingService, TransformerEmbedding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize our embedding service
embedding_service = EmbeddingService()
transformer_model = TransformerEmbedding()
embedding_service.add_model("transformer", transformer_model)

# Pydantic models for request/response validation
class EmbeddingRequest(BaseModel):
    text: str
    model_name: str = "transformer"
    use_cache: bool = True

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: str = "transformer"
    use_cache: bool = True

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    model_name: str = "transformer"
    use_cache: bool = True

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, gt=0, le=100)
    model_name: str = "transformer"

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_name: str
    processing_time: float

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    processing_time: float

class SimilarityResponse(BaseModel):
    similarity: float
    model_name: str
    processing_time: float

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    model_name: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: List[str]

# Initialize FastAPI app
app = FastAPI(
    title="VectorHub API",
    description="Advanced vector search and embedding API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API and available models."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_available=list(embedding_service.models.keys())
    )

# Embedding endpoints
@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Create embedding for a single text."""
    try:
        start_time = time.time()
        
        embedding = embedding_service.get_embedding(
            request.text,
            request.model_name,
            request.use_cache
        )
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embedding=embedding_list,
            model_name=request.model_name,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(request: BatchEmbeddingRequest):
    """Create embeddings for multiple texts."""
    try:
        start_time = time.time()
        
        embeddings = embedding_service.get_embedding(
            request.texts,
            request.model_name,
            request.use_cache
        )
        
        # Convert numpy arrays to lists for JSON serialization
        embeddings_list = embeddings.tolist()
        
        processing_time = time.time() - start_time
        
        return BatchEmbeddingResponse(
            embeddings=embeddings_list,
            model_name=request.model_name,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error creating batch embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute similarity between two texts."""
    try:
        start_time = time.time()
        
        similarity = embedding_service.compute_similarity(
            request.text1,
            request.text2,
            request.model_name,
            request.use_cache
        )
        
        processing_time = time.time() - start_time
        
        return SimilarityResponse(
            similarity=similarity,
            model_name=request.model_name,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Custom exception handler for validation errors
@app.exception_handler(ValueError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)