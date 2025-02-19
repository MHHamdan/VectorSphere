from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.vector_db import vector_db
from app.services.embedding import embedding_service

router = APIRouter()

class VectorInput(BaseModel):
    text: str
    method: str = "sentence-transformers"
    doc_id: int

class VectorQuery(BaseModel):
    text: str
    method: str = "sentence-transformers"
    k: int = 5

@router.post("/add-vector/")
async def add_vector(request: VectorInput):
    """
    API endpoint to add a text's embedding to FAISS.
    """
    try:
        embedding = embedding_service.encode(request.text, request.method)
        vector_db.add_vector(embedding, request.doc_id, request.text)
        return {"message": "Vector added successfully!"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/search-vector/")
async def search_vector(request: VectorQuery):
    """
    API endpoint to search similar embeddings in FAISS.
    """
    try:
        embedding = embedding_service.encode(request.text, request.method)
        results = vector_db.search_vector(embedding, request.k)
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
