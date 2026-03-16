import asyncio
import json
import time

import cv2
import websockets

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.sdp import candidate_from_sdp


SIGNALING_URL = "wss://signaling.ehb.be"
ROOM = "demo"
SELF_ID = "jetson-1"
PEER_ID = "web-1"


def build_signal(msg_type, payload, room=ROOM, from_id=SELF_ID, to_id=PEER_ID):
    return json.dumps({
        "type": msg_type,
        "room": room,
        "from": from_id,
        "to": to_id,
        "payload": payload,
    })


async def consume_video(track, channel_getter):
    """
    Ontvang videoframes, toon ze lokaal, en stuur voorbeeld-heading terug
    via datachannel.
    """
    frame_count = 0
    last_heading_send = 0.0

    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        frame_count += 1

        # Voorbeeld: toon video op Jetson
        cv2.imshow("Jetson WebRTC Receiver", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Dummy heading terugsturen, puur als voorbeeld
        now = time.time()
        if now - last_heading_send > 0.5:
            channel = channel_getter()
            if channel and channel.readyState == "open":
                heading = ((frame_count % 60) - 30) * 1.5
                msg = {
                    "heading": round(heading, 2),
                    "ts": now,
                }
                channel.send(json.dumps(msg))
            last_heading_send = now


async def run():
    pc = RTCPeerConnection(
        RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
    )

    data_channel = {"channel": None}

    def get_channel():
        return data_channel["channel"]

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("PC state:", pc.connectionState)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print("ICE state:", pc.iceConnectionState)

    @pc.on("datachannel")
    def on_datachannel(channel):
        print("DataChannel ontvangen:", channel.label)
        data_channel["channel"] = channel

        @channel.on("open")
        def on_open():
            print("DataChannel open")
            channel.send(json.dumps({
                "message": "Hallo vanuit Jetson"
            }))

        @channel.on("message")
        def on_message(message):
            print("DataChannel bericht:", message)

        @channel.on("close")
        def on_close():
            print("DataChannel gesloten")

    @pc.on("track")
    def on_track(track):
        print("Track ontvangen:", track.kind)
        if track.kind == "video":
            asyncio.create_task(consume_video(track, get_channel))

    async with websockets.connect(SIGNALING_URL) as ws:
        print("Verbonden met signaling:", SIGNALING_URL)

        await ws.send(build_signal(
            "join",
            {"role": "jetson-receiver"}
        ))

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is not None:
                payload = {
                    "candidate": candidate.to_sdp(),
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                }
                await ws.send(build_signal("ice", payload))

        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print("Ongeldig JSON:", raw)
                continue

            if msg.get("room") != ROOM:
                continue
            if msg.get("to") not in (None, SELF_ID):
                continue
            if msg.get("from") == SELF_ID:
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            print("Signal ontvangen:", msg_type)

            if msg_type == "offer":
                offer = RTCSessionDescription(
                    sdp=payload["sdp"],
                    type=payload["type"]
                )
                await pc.setRemoteDescription(offer)

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                await ws.send(build_signal(
                    "answer",
                    {
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type,
                    }
                ))

                print("Answer verzonden")

            elif msg_type == "ice":
                candidate_sdp = payload.get("candidate")
                if candidate_sdp:
                    candidate = candidate_from_sdp(candidate_sdp)
                    candidate.sdpMid = payload.get("sdpMid")
                    candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
                    await pc.addIceCandidate(candidate)
                    print("ICE candidate toegevoegd")

    await pc.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Gestopt door gebruiker")
        cv2.destroyAllWindows()