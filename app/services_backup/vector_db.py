from typing import List, Dict, Any, Optional
import weaviate
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabaseService:
    """Service for managing vector storage and retrieval operations using Weaviate.
    
    This service provides high-level operations for storing and querying vector embeddings,
    with support for batch operations, similarity search, and hybrid search capabilities.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        batch_size: int = 100,
        timeout_config: Optional[Dict[str, int]] = None
    ):
        """Initialize the vector database service.
        
        Args:
            host: Weaviate server host
            port: Weaviate server port
            batch_size: Size of batches for bulk operations
            timeout_config: Optional timeout configuration for operations
        """
        self.batch_size = batch_size
        self.timeout_config = timeout_config or {
            "timeout_retries": 3,
            "timeout_seconds": 300
        }
        
        # Initialize Weaviate client
        try:
            self.client = weaviate.Client(
                f"http://{host}:{port}",
                timeout_config=self.timeout_config
            )
            logger.info("Successfully connected to Weaviate")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise
        
        # Ensure schema exists
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Ensure the required schema exists in Weaviate.
        
        Creates the necessary class definitions if they don't exist.
        """
        class_obj = {
            "class": "Document",
            "vectorizer": "none",  # We provide vectors manually
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The text content associated with the vector",
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata for the document",
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the document was added",
                }
            ]
        }
        
        try:
            if not self.client.schema.contains(class_obj):
                self.client.schema.create_class(class_obj)
                logger.info("Created Document schema in Weaviate")
        except Exception as e:
            logger.error(f"Failed to ensure schema: {str(e)}")
            raise

    def add_document(
        self,
        content: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a single document with its vector embedding to the database.
        
        Args:
            content: The text content of the document
            vector: The vector embedding of the content
            metadata: Optional metadata associated with the document
            
        Returns:
            str: The UUID of the created document
        """
        try:
            # Prepare the data object
            data_object = {
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add the document with its vector
            result = self.client.data_object.create(
                data_object=data_object,
                class_name="Document",
                vector=vector.tolist()
            )
            
            logger.info(f"Successfully added document with UUID: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise

    def add_documents_batch(
        self,
        contents: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add multiple documents with their vector embeddings in batch.
        
        Args:
            contents: List of text contents
            vectors: Array of vector embeddings
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List[str]: List of UUIDs for created documents
        """
        if len(contents) != len(vectors):
            raise ValueError("Number of contents must match number of vectors")
        
        if metadata_list and len(metadata_list) != len(contents):
            raise ValueError("Number of metadata items must match number of contents")
        
        metadata_list = metadata_list or [{}] * len(contents)
        results = []
        
        # Process in batches
        for i in range(0, len(contents), self.batch_size):
            batch_contents = contents[i:i + self.batch_size]
            batch_vectors = vectors[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            
            with self.client.batch as batch:
                batch.batch_size = self.batch_size
                
                try:
                    for content, vector, metadata in zip(batch_contents, batch_vectors, batch_metadata):
                        data_object = {
                            "content": content,
                            "metadata": metadata,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        result = batch.add_data_object(
                            data_object=data_object,
                            class_name="Document",
                            vector=vector.tolist()
                        )
                        results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed during batch processing: {str(e)}")
                    raise
        
        return results

    def search_similar(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity.
        
        Args:
            query_vector: The query vector to search with
            limit: Maximum number of results to return
            metadata_filter: Optional filter conditions for metadata
            
        Returns:
            List of similar documents with their metadata
        """
        try:
            # Prepare the query
            query = self.client.query.get("Document", ["content", "metadata", "timestamp"])
            query = query.with_near_vector({
                "vector": query_vector.tolist()
            })
            
            # Add metadata filters if provided
            if metadata_filter:
                query = query.with_where(metadata_filter)
            
            # Execute search
            result = query.with_limit(limit).do()
            
            # Process and return results
            if "data" in result and "Get" in result["data"]:
                return result["data"]["Get"]["Document"]
            return []
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            raise

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        alpha: float = 0.5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and text search.
        
        Args:
            query_text: Text query for keyword search
            query_vector: Vector query for similarity search
            alpha: Weight between vector and text search (0.0 to 1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of documents ranked by combined score
        """
        try:
            # Prepare the hybrid query
            query = self.client.query.get("Document", ["content", "metadata", "timestamp"])
            
            # Combine vector search with text search
            query = query.with_hybrid(
                query=query_text,
                vector=query_vector.tolist(),
                alpha=alpha
            )
            
            # Execute search
            result = query.with_limit(limit).do()
            
            # Process and return results
            if "data" in result and "Get" in result["data"]:
                return result["data"]["Get"]["Document"]
            return []
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {str(e)}")
            raise

    def delete_document(self, uuid: str) -> bool:
        """Delete a document by UUID.
        
        Args:
            uuid: UUID of the document to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.data_object.delete(
                uuid=uuid,
                class_name="Document"
            )
            logger.info(f"Successfully deleted document with UUID: {uuid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False

    def get_document(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by UUID.
        
        Args:
            uuid: UUID of the document to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Document data if found, None otherwise
        """
        try:
            result = self.client.data_object.get_by_id(
                uuid=uuid,
                class_name="Document"
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve document: {str(e)}")
            return None