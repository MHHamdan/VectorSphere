from sentence_transformers import CrossEncoder

class ReRankingService:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Cross-Encoder for re-ranking search results.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list):
        """
        Re-ranks search results using a BERT-based Cross-Encoder.

        Parameters:
        - query (str): The search query.
        - documents (list): List of retrieved documents from hybrid search.

        Returns:
        - List of re-ranked documents (sorted by relevance).
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort documents by highest relevance score
        ranked_docs = sorted(zip(scores, documents), reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked_docs]

# Singleton instance
reranking_service = ReRankingService()

