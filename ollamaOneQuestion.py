import requests

url = "http://10.2.160.41:11434/api/chat"
payload = {
    "model": "gemma3",
    "messages": [
        {"role": "user", "content": "Ken je Erasmus Hogeschool Brussel?,  Geef me een korte beschrijving van deze school in 1 zin."}
    ],
    "stream": False
}

r = requests.post(url, json=payload, timeout=120)
r.raise_for_status()

data = r.json()
print(data["message"]["content"])