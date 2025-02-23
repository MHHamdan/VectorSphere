from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.redis_producer import redis_producer

router = APIRouter()

class DocumentInput(BaseModel):
    text: str
    doc_id: int

@router.post("/ingest-document/")
async def ingest_document(request: DocumentInput):
    """
    API endpoint to add a document to the Redis Stream.
    """
    try:
        response = redis_producer.add_document(request.doc_id, request.text)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

