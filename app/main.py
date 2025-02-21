from fastapi import FastAPI
from app.routes import embeddings, vector_db, search, hybrid_search

# Initialize FastAPI app
app = FastAPI(title="VectorSphere", description="AI-powered Hybrid Search System")

# Include routes
app.include_router(embeddings.router, prefix="/api", tags=["Embeddings"])
app.include_router(vector_db.router, prefix="/api", tags=["Vector Database"])
app.include_router(search.router, prefix="/api", tags=["Vector Search"])
app.include_router(hybrid_search.router, prefix="/api", tags=["Hybrid Search"])

@app.get("/")
def health_check():
    """ Health check endpoint to verify the API is running. """
    return {"status": "API is running successfully 🚀"}

