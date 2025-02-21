from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.hybrid_search import hybrid_search

router = APIRouter()

class HybridSearchRequest(BaseModel):
    query: str
    k: int = 5

class DocumentInput(BaseModel):
    text: str
    doc_id: int

class BulkDocumentsInput(BaseModel):
    documents: List[DocumentInput]

@router.post("/add-document/")
async def add_document(request: DocumentInput):
    """
    API endpoint to add a document for hybrid search.
    """
    try:
        hybrid_search.add_document(request.text, request.doc_id)
        return {"message": "Document added successfully! Processing in background..."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/bulk-add-documents/")
async def bulk_add_documents(request: BulkDocumentsInput):
    """
    API endpoint to add multiple documents at once.
    """
    try:
        for doc in request.documents:
            hybrid_search.add_document(doc.text, doc.doc_id)
        return {"message": "Bulk documents added successfully! Processing in background..."}
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

