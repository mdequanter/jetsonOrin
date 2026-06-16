import asyncio
import websockets
import json
import argparse

DEFAULT_SERVER = "ws://jetson-desktop:9000"
DEFAULT_ROOM = "/ws/testroom"

async def receive_messages(server, room):

    uri = server.rstrip("/") + room

    print("🔌 Connecting to signaling server...")
    print("Server :", server)
    print("Room   :", room)

    async with websockets.connect(
        uri,
        compression=None,
    ) as websocket:

        print(f"✅ Connected to signaling server ({uri})")

        while True:
            try:
                message = await websocket.recv()
                print("📩 Raw message received:", message)

                data = json.loads(message)

                print("📦 Parsed JSON:")
                print("   Type :", data.get("type"))
                print("   From :", data.get("from"))
                print("   Data :", data.get("data"))
                print("-" * 40)

            except websockets.exceptions.ConnectionClosed:
                print("⚠ Connection to server closed.")
                break
            except json.JSONDecodeError:
                print("❌ Could not parse JSON message.")


def main():

    parser = argparse.ArgumentParser(description="WebSocket signaling receiver")

    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Signaling server (default: {DEFAULT_SERVER})"
    )

    parser.add_argument(
        "--room",
        default=DEFAULT_ROOM,
        help=f"Room path (default: {DEFAULT_ROOM})"
    )

    args = parser.parse_args()

    asyncio.run(receive_messages(args.server, args.room))


if __name__ == "__main__":
    main()
