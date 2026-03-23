import requests

url = "http://10.2.160.41:11434/api/chat"

# volledige chatgeschiedenis
messages = []

while True:
    user_input = input("Jij: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    # voeg user bericht toe
    messages.append({
        "role": "user",
        "content": user_input
    })

    payload = {
        "model": "gemma3",
        "messages": messages,
        "stream": False
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    assistant_reply = data["message"]["content"]

    print("AI:", assistant_reply)

    # voeg antwoord toe aan geschiedenis
    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })