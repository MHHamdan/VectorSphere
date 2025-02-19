from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.embedding import embedding_service

router = APIRouter()

# Request body model
class EmbeddingRequest(BaseModel):
    text: str
    method: str = "sentence-transformers"

@router.post("/generate-embedding/")
async def generate_embedding(request: EmbeddingRequest):
    """
    API endpoint to generate embeddings from input text.
    """
    try:
        embedding = embedding_service.encode(request.text, request.method)
        return {"embedding": embedding}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

