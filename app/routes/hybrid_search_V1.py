from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.hybrid_search import hybrid_search

router = APIRouter()

class HybridSearchRequest(BaseModel):
    query: str
    k: int = 5

class DocumentInput(BaseModel):
    text: str
    doc_id: int

@router.post("/add-document/")
async def add_document(request: DocumentInput):
    """
    API endpoint to add a document for hybrid search.
    """
    try:
        hybrid_search.add_document(request.text, request.doc_id)
        return {"message": "Document added successfully!"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/hybrid-search/")
async def search(request: HybridSearchRequest):
    """
    API endpoint to perform hybrid search.
    """
    try:
        results = hybrid_search.search(request.query, request.k)
        return {"query": request.query, "results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

