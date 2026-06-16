import asyncio
import json
import websockets
import time

ROOM = "/ws/testroom"
SIGNALING_SERVER = f"ws://jetson-desktop:9000{ROOM}"
SESSION_ID = "demo-session-001"

async def send_message():

    message = {
        "sessionId": SESSION_ID,
        "type": "topic",
        "from": "client1",
        "to": "receiver1", # or "all"
        "data": {
            "name": "topic1",
            "value": time.time()
        }
    }

    uri = SIGNALING_SERVER  # Zelfde server als sender
    async with websockets.connect(uri,
        compression=None,
    ) as websocket:

        print("Connected")

        await websocket.send(json.dumps(message))
        print("Message sent")

        # wacht op antwoorden
        #async for msg in websocket:
        #    print("Received:", msg)


asyncio.run(send_message())
