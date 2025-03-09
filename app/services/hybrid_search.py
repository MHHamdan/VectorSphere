from rank_bm25 import BM25Okapi
from app.services.vector_db import vector_db
from app.services.embedding import embedding_service
import numpy as np
from collections import deque
import threading
import time
from app.services.reranking import reranking_service
from app.services.query_expansion import query_expansion
from app.services.learning_to_rank import learning_to_rank  # Import LTR


class HybridSearch:
    def __init__(self):
        """
        Initializes the hybrid search engine with BM25 and FAISS vector search.
        Uses a queue to optimize real-time indexing.
        """
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.index_queue = deque()  # Queue for batch processing
        self.lock = threading.Lock()

        # Start background indexing thread
        self.indexing_thread = threading.Thread(target=self._process_queue)
        self.indexing_thread.daemon = True
        self.indexing_thread.start()

    def _process_queue(self):
        """
        Processes the queue to batch index new documents every 5 seconds.
        """
        while True:
            time.sleep(5)  # Batch processing interval
            if self.index_queue:
                with self.lock:
                    batch = list(self.index_queue)
                    self.index_queue.clear()

                    for doc_id, text in batch:
                        tokens = text.lower().split()  # Simple tokenization
                        self.tokenized_corpus.append(tokens)
                        self.documents.append((doc_id, text))

                        # Update BM25 index
                        self.bm25 = BM25Okapi(self.tokenized_corpus)

                        # Generate and store vector embeddings
                        embedding = embedding_service.encode(text, "sentence-transformers")
                        faiss.normalize_L2(embedding)  # Normalize for better similarity
                        vector_db.add_vector(embedding, doc_id, text)




    def add_document(self, text: str, doc_id: int):
        """
        Adds a document to the indexing queue for real-time processing.
        """
        with self.lock:
            self.index_queue.append((doc_id, text))

    def search(self, query: str, k=5):
        """
        Performs hybrid search using BM25, vector search, and re-ranking.
        """
        # Expand query
        expanded_queries = query_expansion.expand_query(query)
        
        fusion_results = {}

        
        for expanded_query in expanded_queries:
            query_vector = embedding_service.encode(expanded_query, "sentence-transformers")
            
            # BM25 Search
            bm25_results = self.bm25_search(expanded_query, k)
            tokenized_query = expanded_query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # FAISS Vector Search
            vector_results = vector_db.search_vector(query_vector, k)

            # Combine Results Using Weighted Fusion
            for doc, score in zip(bm25_results, bm25_scores):
                fusion_results[doc] = fusion_results.get(doc, 0) + score * 0.7  # BM25 weight

            for doc in vector_results:
                fusion_results[doc] = fusion_results.get(doc, 0) + 0.3  # Vector weight

        # Re-rank final results
        query_vector = embedding_service.encode(query, "sentence-transformers")
        ranked_results = learning_to_rank.predict_ranking(query_vector, list(fusion_results.keys()))

        return ranked_results[:k]

# Singleton instance
hybrid_search = HybridSearch()

