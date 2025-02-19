import faiss
import numpy as np

class VectorDB:
    def __init__(self, vector_dim=384):
        """
        Initializes the FAISS vector index.
        
        Parameters:
        - vector_dim (int): Dimension of the embedding vectors.
        """
        self.index = faiss.IndexFlatL2(vector_dim)  # L2 distance (Euclidean)
        self.id_to_text = {}  # Mapping for ID to original text

    def add_vector(self, vector: list, doc_id: int, text: str):
        """
        Adds a vector to the FAISS index.
        
        Parameters:
        - vector (list): The embedding vector.
        - doc_id (int): Unique ID for the document.
        - text (str): Original text (for retrieval).
        """
        vector = np.array([vector]).astype('float32')
        self.index.add(vector)
        self.id_to_text[doc_id] = text

    def search_vector(self, query_vector: list, k=5):
        """
        Searches for similar vectors in the FAISS index.
        
        Parameters:
        - query_vector (list): Query embedding vector.
        - k (int): Number of nearest neighbors to return.
        
        Returns:
        - List of closest document texts.
        """
        query_vector = np.array([query_vector]).astype('float32')
        _, indices = self.index.search(query_vector, k)
        return [self.id_to_text[idx] for idx in indices[0] if idx in self.id_to_text]

# Singleton instance
vector_db = VectorDB()

