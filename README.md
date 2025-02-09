<div align="center">
  <img src="assets/images/vectorsphere-logo.png" alt="VectorSphere Logo" width="200"/>
</div>

# VectorSphere: Advanced Vector Search & Retrieval System

## Project Overview

VectorSphere is a production-ready vector search system that demonstrates state-of-the-art approaches to semantic search and retrieval at scale. The system implements multiple embedding models, vector search algorithms, and retrieval approaches to provide flexible and powerful semantic search capabilities.

### Key Features

- **Multi-model Embedding Pipeline**
  - Support for various embedding models (e.g., SentenceTransformers, OpenAI embeddings)
  - Model fine-tuning capabilities
  - Embedding caching and optimization
- **Scalable Vector Search Implementation**
  - HNSW algorithm integration [(Reference)](https://www.pinecone.io/learn/series/faiss/hnsw/)
  - Faiss-based optimization [(Reference)](https://www.pinecone.io/learn/series/faiss/)
  - Efficient similarity computation
- **Advanced Retrieval System**
  - Hybrid search combining BM25 and vector similarity [(Reference)](https://weaviate.io/blog/hybrid-search-explained)
  - Sophisticated re-ranking pipeline
  - RAG (Retrieval-Augmented Generation) with configurable LLM backends
- **Production-Ready Infrastructure**
  - Comprehensive MLOps pipeline for model deployment and monitoring
  - Robust observability with Prometheus and Grafana
  - Performance benchmarking tools

## Technical Architecture

### Core Components

1. **Embedding Service**

   - Multiple embedding model support (SentenceTransformers, OpenAI embeddings)
   - Fine-tuning pipeline
   - Caching and optimization

2. **Vector Storage & Search**

   - Primary: Weaviate integration for scalable vector storage
   - Secondary: Faiss for rapid prototyping
   - PGVector hybrid search capabilities

3. **Retrieval System**

   - Hybrid search combining vector and text approaches
   - Re-ranking pipeline
   - RAG implementation with LLM backends

4. **API Layer**

   - FastAPI implementation with async support
   - Comprehensive API documentation
   - Rate limiting and security mechanisms

5. **MLOps Pipeline**
   - Docker containerization
   - Kubeflow pipelines for training and deployment
   - Monitoring and observability (Prometheus, Grafana)

## Technology Stack

- **Backend Framework**: FastAPI
- **Machine Learning**: PyTorch, transformers
- **Vector Search**: Weaviate, Faiss, PGVector
- **Traditional Search**: Elasticsearch
- **Data Processing**: Ray, Dask
- **MLOps**: Docker, Kubeflow
- **Monitoring**: Prometheus, Grafana

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.9 or higher
- Docker and Docker Compose
- Git

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/MHHamdan/VectorSphere.git
<<<<<<< HEAD
=======
cd VectorSphere
>>>>>>> 5c87cc1 (Add configuration management with environment var)

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Start the services
docker-compose up -d

# Run the API
uvicorn app.main:app --reload
```

### Accessing the Services

- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Vector Database Console: [http://localhost:8080](http://localhost:8080)

## Project Structure

```
VectorSphere/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py        # Embedding service implementation
│   │   └── vector_db.py        # Vector database service
│   └── core/
│       ├── __init__.py
│       └── config.py           # Configuration module
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Test configurations and fixtures
│   ├── test_embedding_service.py
│   ├── test_vector_db_service.py
│   └── test_api.py
│
├── Dockerfile                  # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore             # Docker ignore file
├── requirements.txt          # Project dependencies
└── pytest.ini               # Pytest configuration
```

## Development Roadmap

1. **Core Implementation**

   - Base embedding functionality
   - Vector search system
   - REST API development

2. **Advanced Features**

   - Hybrid search implementation
   - RAG integration
   - Performance optimization

3. **Production Readiness**
   - MLOps pipeline
   - Monitoring and observability
   - Documentation and examples

## Contributing

We welcome contributions that align with our project goals. Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or suggestions, please open an issue in the GitHub repository: [https://github.com/MHHamdan/VectorSphere](https://github.com/MHHamdan/VectorSphere)
