import requests

url = "http://127.0.0.1:8000/embed"
data = {"text": "This is a test sentence"}

response = requests.post(url, json=data)
print(response.json())
