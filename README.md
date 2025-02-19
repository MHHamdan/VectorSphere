<h1>🚀 VectorSphere: Scalable AI-Powered Vector Search System</h1>

<p align="center">
  <img src="https://github.com/MHHamdan/VectorSphere/blob/main/assets/images/vectorsphere-logo.png" alt="VectorSphere Logo" width="250">
</p>

## 📌 Overview


VectorSphere is a **high-performance AI-powered vector search system** designed for **semantic search and retrieval at scale**. It integrates **multiple embedding models, advanced vector search algorithms, and retrieval techniques**, making it ideal for **recommendation systems, intelligent search, and knowledge retrieval**.

---

## ✨ Key Features

### 🔹 Multi-Model Embedding Pipeline
- ✅ Supports **multiple embedding models**: `SentenceTransformers`, `OpenAI`, `Hugging Face`, `Custom Fine-tuned Models`
- ✅ **Optimized embedding caching** for efficiency
- ✅ **Supports fine-tuning models** for domain-specific embeddings

### 🔹 Scalable Vector Search Implementation
- ✅ **HNSW algorithm** for fast approximate nearest neighbor search
- ✅ **FAISS-based indexing** for scalable retrieval
- ✅ **Hybrid Search** (BM25 + Vector Similarity) for improved results

### 🔹 Advanced Retrieval System
- ✅ **Combines BM25 text search with vector similarity**
- ✅ **Re-ranking pipeline** to refine search results
- ✅ **Retrieval-Augmented Generation (RAG)** using LLMs

### 🔹 Production-Ready Infrastructure
- ✅ **MLOps pipeline** for model deployment & monitoring
- ✅ **Observability with Prometheus & Grafana**
- ✅ **Performance benchmarking tools**

---

## ⚙️ Technical Architecture

VectorSphere follows a **modular and scalable architecture**:

### 📌 Core Components

#### 🔹 **Embedding Service**
- **Multiple embedding model support** (`SentenceTransformers`, `OpenAI`, `Custom Models`)
- **Fine-tuning pipeline** for adapting embeddings to specific domains
- **Caching and optimization** for efficient query processing

#### 🔹 **Vector Storage & Search**
- **Primary:** `Weaviate` integration for scalable vector storage
- **Secondary:** `FAISS` for rapid prototyping and experimentation
- **Hybrid:** `PGVector` for vector storage inside `PostgreSQL`

#### 🔹 **Retrieval System**
- **Hybrid search**: combines traditional **BM25** text search with **vector similarity**
- **Re-ranking pipeline**: uses AI models to improve search relevance
- **Retrieval-Augmented Generation (RAG)** with LLM backends

#### 🔹 **API Layer**
- **Built with FastAPI** for high-performance APIs
- **Async support** for handling high-load requests
- **Security mechanisms** (rate limiting, authentication)

#### 🔹 **MLOps Pipeline**
- **Containerized using Docker**
- **Kubeflow pipelines** for model training & deployment
- **Monitoring & Observability** with `Prometheus` and `Grafana`

---

## 🛠 Technology Stack

| **Category**        | **Technology Used** |
|---------------------|--------------------|
| **Backend Framework** | FastAPI |
| **Machine Learning** | PyTorch, Hugging Face Transformers |
| **Vector Search** | Weaviate, FAISS, PGVector |
| **Text Search** | Elasticsearch |
| **Data Processing** | Ray, Dask |
| **MLOps** | Docker, Kubeflow |
| **Monitoring** | Prometheus, Grafana |

---

## 🔧 Getting Started

### 📌 Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.9+**
- **Docker & Docker Compose**
- **Git**

### 📌 Installation Steps

```bash
# Clone the repository
git clone https://github.com/MHHamdan/VectorSphere.git
cd VectorSphere

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## 📌 Running the Project

```bash
# Start the services
docker-compose up -d

# Run the API
uvicorn app.main:app --reload
```

## 🌍 Accessing the Services

| **Service**                | **URL**                              |
|----------------------------|--------------------------------------|
| **API Documentation**      | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Vector Database Console** | [http://localhost:8080](http://localhost:8080) |

## 📂 Project Structure

```graphql
VectorSphere/
│
├── app/
│   ├── main.py                 # FastAPI application
│   ├── services/
│   │   ├── embedding.py        # Embedding service
│   │   ├── vector_db.py        # Vector database service
│   │   ├── hybrid_search.py    # BM25 + Vector hybrid search
│   │   ├── reranking.py        # Re-ranking module
│   │   └── streaming.py        # Real-time vector ingestion
│   ├── core/
│   │   ├── config.py           # Configuration module
│   │   ├── dependencies.py     # API dependencies
│   └── routes/
│       ├── search.py           # API endpoints for search
│       ├── embeddings.py       # API endpoints for embeddings
│       ├── health.py           # Health check routes
│
├── tests/
│   ├── test_embedding_service.py
│   ├── test_vector_db_service.py
│   ├── test_api.py
│
├── scripts/                    # Utility scripts
│   ├── benchmark.py            # Benchmarking FAISS vs Weaviate
│   ├── ingest_vectors.py       # Vector ingestion
│   ├── update_index.py         # Refresh FAISS index
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .env
```

## 🚀 Development Roadmap

```yaml
✅ Phase 1: Core Implementation
  - Base embedding functionality
  - Vector search system
  - REST API

🔜 Phase 2: Advanced Features
  - Hybrid search implementation (BM25 + Vectors)
  - RAG integration
  - Performance optimization

🔜 Phase 3: Production Readiness
  - MLOps pipeline
  - Monitoring & observability
  - Documentation and examples
```

## 🤝 Contributing

We welcome contributions! Please check out our **[Contributor Guide](https://github.com/MHHamdan/VectorSphere/blob/main/CONTRIBUTING.md)** before submitting pull requests.

## 📜 License

This project is licensed under the **MIT License**.

## 📩 Contact

For questions or suggestions, open an **issue** on GitHub:  
🔗 [VectorSphere GitHub Repo](https://github.com/MHHamdan/VectorSphere)




