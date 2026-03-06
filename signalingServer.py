import asyncio
import websockets
from collections import defaultdict

# room_id -> set(websocket)
rooms = defaultdict(set)
rooms_lock = asyncio.Lock()


def room_from_path(path: str) -> str:
    """
    Verwacht paden zoals:
      /ws/demo-session-001
      /room/demo
    """

    print (f"Received connection with path: {path}")
    path = (path or "/").split("?", 1)[0]
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("ws", "room"):
        return parts[1]
    # fallback: alles in default room
    return "default"


async def signaling(websocket):
    # websockets >= 12: path zit in websocket.request.path
    path = websocket.request.path if hasattr(websocket, "request") else "/"
    room_id = room_from_path(path)

    async with rooms_lock:
        rooms[room_id].add(websocket)
        peer_count = len(rooms[room_id])

    # (optioneel) informeer de client
    try:
        await websocket.send(f'{{"type":"room_joined","room":"{room_id}","peers":{peer_count}}}')
    except Exception:
        pass

    print(f"✅ Client joined room '{room_id}' from {websocket.remote_address} (peers={peer_count})")

    try:
        async for message in websocket:
            # Forward ONLY to peers in same room
            async with rooms_lock:
                targets = [ws for ws in rooms[room_id] if ws != websocket]

            # kleine optimalisatie: gather
            if targets:
                await asyncio.gather(*(t.send(message) for t in targets), return_exceptions=True)

    finally:
        async with rooms_lock:
            rooms[room_id].discard(websocket)
            remaining = len(rooms[room_id])
            if remaining == 0:
                rooms.pop(room_id, None)

        print(f"❌ Client left room '{room_id}' ({websocket.remote_address}); remaining={remaining}")


async def start_server():
    print("🚀 Room signaling server on ws://0.0.0.0:9000  (paths: /ws/<room>)")
    async with websockets.serve(
        signaling,
        "0.0.0.0",
        9000,
        max_size=2**23,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(start_server())
