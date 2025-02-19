from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI(title="VectorSphere", description="AI-powered Vector Search System")

@app.get("/")
def health_check():
    """ Health check endpoint to verify the API is running. """
    return {"status": "API is running successfully 🚀"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

