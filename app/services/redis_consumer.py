import redis
import os
import time
from app.services.hybrid_search import hybrid_search

class RedisConsumer:
    def __init__(self):
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)

    def process_documents(self):
        """
        Continuously listens to Redis Stream and indexes new documents in real-time.
        """
        while True:
            try:
                stream_data = self.redis_client.xread({"document_stream": "$"}, block=0, count=1)
                if stream_data:
                    _, entries = stream_data[0]
                    for entry_id, data in entries:
                        doc_id = int(data["doc_id"])
                        #text = data["text"]
			doc_id = int(data["doc_id"])
			text = self.redis_client.hget(f"doc:{doc_id}", "text")

			if text:  # Only process if text exists
			    hybrid_search.add_document(text, doc_id)
			    print(f"Indexed document {doc_id}: {text}")
	
                        hybrid_search.add_document(text, doc_id)
                        print(f"Indexed document {doc_id}: {text}")
            except Exception as e:
                print(f"Error processing Redis Stream: {e}")
            time.sleep(1)

# Start the Redis Consumer
if __name__ == "__main__":
    consumer = RedisConsumer()
    consumer.process_documents()

