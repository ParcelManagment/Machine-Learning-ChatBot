import requests

url = "http://127.0.0.1:8000/chat"
headers = {"Content-Type": "application/json"}
data = {"message": "Hello, how are you?"}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("Bot:", response.json()["response"])
else:
    print("Error:", response.status_code, response.text)