import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.embedding import EmbeddingService, TransformerEmbedding
from app.services.vector_db import VectorDatabaseService
import numpy as np

@pytest.fixture
def test_client():
    """Create a test client for our FastAPI app."""
    return TestClient(app)

@pytest.fixture
def embedding_service():
    """Create a test embedding service."""
    service = EmbeddingService()
    model = TransformerEmbedding()
    service.add_model("test-transformer", model)
    return service

@pytest.fixture
def vector_db_service():
    """Create a test vector database service."""
    return VectorDatabaseService(
        host="localhost",
        port=8080,
        batch_size=10
    )

@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "This is a test document for vector search.",
        "Another example document with different content.",
        "A third document to test batch processing."
    ]

@pytest.fixture
def sample_vectors():
    """Provide sample vectors for testing."""
    return np.random.rand(3, 384)  # 384 is the dimension of our embeddings

# tests/test_embedding_service.py
import pytest
import numpy as np
from app.services.embedding import TransformerEmbedding

def test_transformer_embedding_initialization():
    """Test that the transformer embedding model initializes correctly."""
    model = TransformerEmbedding()
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None

def test_embedding_generation(embedding_service, sample_texts):
    """Test that embeddings are generated correctly."""
    # Get embeddings for sample texts
    embeddings = embedding_service.get_embedding(
        sample_texts[0],
        "test-transformer"
    )
    
    # Check embedding shape and properties
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 1
    assert embeddings.shape[0] == 384  # Expected embedding dimension
    
    # Check that embeddings are normalized
    norm = np.linalg.norm(embeddings)
    assert np.isclose(norm, 1.0, atol=1e-6)

def test_batch_embedding_generation(embedding_service, sample_texts):
    """Test batch embedding generation."""
    # Generate embeddings for multiple texts
    embeddings = embedding_service.get_embedding(
        sample_texts,
        "test-transformer"
    )
    
    # Check batch embedding properties
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 2
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] == 384  # Expected embedding dimension

def test_similarity_computation(embedding_service):
    """Test similarity computation between texts."""
    text1 = "This is a test sentence."
    text2 = "This is also a test sentence."
    text3 = "This is completely different content."
    
    # Compute similarities
    similarity_close = embedding_service.compute_similarity(
        text1,
        text2,
        "test-transformer"
    )
    similarity_far = embedding_service.compute_similarity(
        text1,
        text3,
        "test-transformer"
    )
    
    # Check that similar texts have higher similarity
    assert similarity_close > similarity_far
    assert 0 <= similarity_close <= 1
    assert 0 <= similarity_far <= 1

# tests/test_vector_db_service.py
import pytest
import numpy as np
from app.services.vector_db import VectorDatabaseService

def test_vector_db_initialization(vector_db_service):
    """Test vector database initialization."""
    assert vector_db_service is not None
    assert vector_db_service.client is not None

def test_document_addition(vector_db_service):
    """Test adding a single document to the vector database."""
    # Create test document
    content = "Test document content"
    vector = np.random.rand(384)
    metadata = {"source": "test"}
    
    # Add document
    doc_id = vector_db_service.add_document(content, vector, metadata)
    
    # Verify document was added
    assert doc_id is not None
    
    # Retrieve and verify document
    doc = vector_db_service.get_document(doc_id)
    assert doc is not None
    assert doc["content"] == content
    assert doc["metadata"]["source"] == "test"

def test_batch_document_addition(vector_db_service, sample_texts, sample_vectors):
    """Test adding multiple documents in batch."""
    metadata_list = [{"source": "test"} for _ in sample_texts]
    
    # Add documents in batch
    doc_ids = vector_db_service.add_documents_batch(
        sample_texts,
        sample_vectors,
        metadata_list
    )
    
    # Verify all documents were added
    assert len(doc_ids) == len(sample_texts)
    
    # Verify each document
    for doc_id, expected_content in zip(doc_ids, sample_texts):
        doc = vector_db_service.get_document(doc_id)
        assert doc is not None
        assert doc["content"] == expected_content

def test_similarity_search(vector_db_service, sample_texts, sample_vectors):
    """Test similarity search functionality."""
    # Add documents to search
    vector_db_service.add_documents_batch(sample_texts, sample_vectors)
    
    # Perform similarity search
    query_vector = sample_vectors[0]
    results = vector_db_service.search_similar(query_vector, limit=2)
    
    # Verify search results
    assert len(results) <= 2
    assert results[0]["content"] in sample_texts

# tests/test_api.py
from fastapi.testclient import TestClient
import numpy as np

def test_health_endpoint(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_embedding_endpoint(test_client):
    """Test the embedding generation endpoint."""
    payload = {
        "text": "Test document for embedding generation.",
        "model_name": "transformer",
        "use_cache": True
    }
    
    response = test_client.post("/embed", json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert "embedding" in result
    assert "processing_time" in result
    assert len(result["embedding"]) == 384  # Expected embedding dimension

def test_batch_embedding_endpoint(test_client):
    """Test the batch embedding generation endpoint."""
    payload = {
        "texts": [
            "First test document.",
            "Second test document.",
            "Third test document."
        ],
        "model_name": "transformer",
        "use_cache": True
    }
    
    response = test_client.post("/embed/batch", json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert "embeddings" in result
    assert len(result["embeddings"]) == len(payload["texts"])
    assert len(result["embeddings"][0]) == 384  # Expected embedding dimension

# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests