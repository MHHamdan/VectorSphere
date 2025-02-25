import openai
import os
from transformers import pipeline

class QueryExpansion:
    def __init__(self, method="openai"):
        """
        Initializes the query expansion service with an LLM.
        Supports OpenAI API and Hugging Face Transformers.
        """
        self.method = method
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.hf_model = pipeline("text-generation", model="facebook/bart-large-cnn")

    def expand_query(self, query: str):
        """
        Expands the search query using an LLM.

        Parameters:
        - query (str): The original user query.

        Returns:
        - List[str]: Expanded queries.
        """
        if self.method == "openai":
            if not self.openai_api_key:
                raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY environment variable.")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Expand the following query with synonyms and alternative phrasing."},
                          {"role": "user", "content": query}]
            )
            expanded_query = response["choices"][0]["message"]["content"]
            return [query] + expanded_query.split("\n")

        elif self.method == "huggingface":
            result = self.hf_model(f"Expand this query: {query}", max_length=50, num_return_sequences=1)
            return [query] + [result[0]["generated_text"]]

        else:
            raise ValueError("Invalid expansion method. Use 'openai' or 'huggingface'.")

# Singleton instance
query_expansion = QueryExpansion()

