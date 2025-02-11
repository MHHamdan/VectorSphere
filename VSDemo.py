import requests
import json

def test_vectorsphere_embedding():
    """Demo function to test VectorSphere's embedding capabilities."""
    api_url = "http://localhost:8000/embed"

    # Example text input for embedding
    payload = {
        "text": "This is a test sentence for embedding."
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=payload)

    # Check response
    if response.status_code == 200:
        result = response.json()
        print("Embedding Result:", json.dumps(result, indent=4))
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_vectorsphere_embedding()
