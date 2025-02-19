from sentence_transformers import SentenceTransformer
import openai
import os

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the embedding service.
        Supports SentenceTransformers and OpenAI embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def encode(self, text: str, method: str = "sentence-transformers"):
        """
        Generates embeddings from text using the selected method.

        Parameters:
        - text (str): Input text to encode.
        - method (str): Embedding model type ("sentence-transformers" or "openai").

        Returns:
        - List[float]: Generated embedding vector.
        """
        if method == "sentence-transformers":
            return self.model.encode(text).tolist()
        elif method == "openai":
            if not self.openai_api_key:
                raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY environment variable.")
            return openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
        else:
            raise ValueError("Invalid embedding method. Use 'sentence-transformers' or 'openai'.")

# Create a singleton instance of the embedding service
embedding_service = EmbeddingService()


