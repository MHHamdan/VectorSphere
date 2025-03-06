from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.embedding import embedding_service
from app.services.vector_db import vector_db
from hybrid_search import hybrid_search
from fastapi import Depends
from app.services.auth import decode_access_token

search_history = {}

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    method: str = "sentence-transformers"
    k: int = 5

# @router.post("/search/")
# async def search(request: SearchRequest):
#     """
#     API endpoint to search for relevant vectors based on a query.
#     """
#     try:
#         query_embedding = embedding_service.encode(request.query, request.method)
#         results = vector_db.search_vector(query_embedding, request.k)
#         return {"query": request.query, "results": results}
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))


@router.post("/search/")
async def search(request: SearchRequest, token: str = Depends()):
    """
    Search API with user history tracking.
    """
    user_data = decode_access_token(token)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid token")

    username = user_data["sub"]

    # Store user search history
    if username not in search_history:
        search_history[username] = []
    search_history[username].append(request.query)

    results = hybrid_search.search(request.query, request.k)
    return {"query": request.query, "results": results, "history": search_history[username]}