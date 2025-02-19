from fastapi import FastAPI
from app.routes import embeddings, vector_db

# Initialize FastAPI app
app = FastAPI(title="VectorSphere", description="AI-powered Vector Search System")

# Include routes
app.include_router(embeddings.router, prefix="/api", tags=["Embeddings"])
app.include_router(vector_db.router, prefix="/api", tags=["Vector Database"])

@app.get("/")
def health_check():
    """ Health check endpoint to verify the API is running. """
    return {"status": "API is running successfully 🚀"}

