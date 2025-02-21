from rank_bm25 import BM25Okapi
from app.services.vector_db import vector_db
from app.services.embedding import embedding_service
import numpy as np

class HybridSearch:
    def __init__(self):
        """
        Initializes the hybrid search engine with BM25 and FAISS vector search.
        """
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

    def add_document(self, text: str, doc_id: int):
        """
        Adds a document to both BM25 and FAISS vector storage.
        """
        self.documents.append((doc_id, text))
        tokens = text.lower().split()  # Simple tokenization
        self.tokenized_corpus.append(tokens)
        
        # Update BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Generate and store vector embeddings
        embedding = embedding_service.encode(text, "sentence-transformers")
        vector_db.add_vector(embedding, doc_id, text)

    def search(self, query: str, k=5):
        """
        Performs hybrid search using BM25 and vector similarity.
        """
        query_vector = embedding_service.encode(query, "sentence-transformers")

        # BM25 text-based search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_k_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [self.documents[idx][1] for idx in bm25_top_k_indices if idx < len(self.documents)]

        # FAISS vector search
        vector_results = vector_db.search_vector(query_vector, k)

        # Combine results (Hybrid approach)
        combined_results = list(set(bm25_results + vector_results))[:k]  # Merge results
        return combined_results

# Singleton instance
hybrid_search = HybridSearch()

