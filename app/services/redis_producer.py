import redis
import os
import json

class RedisProducer:
    def __init__(self):
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)

    def add_document(self, doc_id: int, text: str):
        """
        Adds a new document to the Redis Stream for processing.
        """
        data = {"doc_id": doc_id, "text": text}
        self.redis_client.xadd("document_stream", data)
        return {"message": "Document added to Redis Stream for processing"}

# Singleton instance
redis_producer = RedisProducer()

