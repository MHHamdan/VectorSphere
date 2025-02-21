from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.embedding import embedding_service
from app.services.vector_db import vector_db

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    method: str = "sentence-transformers"
    k: int = 5

@router.post("/search/")
async def search(request: SearchRequest):
    """
    API endpoint to search for relevant vectors based on a query.
    """
    try:
        query_embedding = embedding_service.encode(request.query, request.method)
        results = vector_db.search_vector(query_embedding, request.k)
        return {"query": request.query, "results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

