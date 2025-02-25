from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.query_expansion import query_expansion

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    method: str = "openai"

@router.post("/expand-query/")
async def expand_query(request: QueryRequest):
    """
    API endpoint to expand search queries.
    """
    try:
        expanded_queries = query_expansion.expand_query(request.query)
        return {"original_query": request.query, "expanded_queries": expanded_queries}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

