import base64
import requests

OLLAMA_URL = "http://10.2.160.41:11434/api/chat"
IMAGE_PATH = "static\\images\\foto1.jpg"

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")


#print (image_b64)
payload = {
    "model": "llama3.2-vision:11b",
    "messages": [
        {
            "role": "user",
            "content": "Beschrijf deze foto in het Nederlands heel kort in 1 zin.",
            "images": [image_b64]
        }
    ],
    "stream": False
}

r = requests.post(OLLAMA_URL, json=payload, timeout=120)
r.raise_for_status()

data = r.json()
print(data["message"]["content"])