import asyncio
import logging
import json
import sys
import tty
import termios
import threading
import random
import time

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD, VUI_COLOR

logging.basicConfig(level=logging.FATAL)

MOVE_SPEED = 0.5
TURN_SPEED = 1
BRIGHTNESS_LVL = 1
DISCO_COLOURS = [VUI_COLOR.RED, VUI_COLOR.YELLOW, VUI_COLOR.GREEN, VUI_COLOR.CYAN, VUI_COLOR.BLUE, VUI_COLOR.PURPLE]

def print_controls():
    print("""
================= CONTROLS =================
   THE BASICS
-----------------
z : vooruit
s : achteruit
a : draaien links
e : draaien rechts
q : zijwaarts links
d : zijwaarts rechts
o : opstaan
l : neerliggen
-----------------
 EXTRA MOVEMENTS
-----------------
h : hello
x : stretch
y : Sit / t: Rize sit
f : Wiggle

m : moonwalk (NON FUNCTIONAL)
-----------------
 LIGHT CONTROLS
-----------------
u : zaklamp uit
i : zaklamp aan
    1. White        🤍
    2. Red          ❤️
    3. Yellow       💛
    4. Green        💚
    5. Cyan         🩵
    6. Blue         💙 
    7. Purple       💜 
    8. Disco        🪩 
    9. Flash        🔦 
    0. Licht uit    🖤
-----------------
  HELP CONTROLS
-----------------
r : EMERGENCY STOP
p : print controls
(ctrl + c) * 2 : exit
============================================
""")

async def send_move(conn, x=0, y=0, z=0):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z}
        }
    )

def get_key():
    """Read a single keypress from stdin (works over SSH)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_listener(conn, loop, stop_event):
    """Blocking keyboard loop running in a separate thread."""
    while not stop_event.is_set():
        try:
            k = get_key().lower()
        except Exception:
            break

        # Ctrl+C
        if k == "\x03":
            print("\nExit requested")
            stop_event.set()
            break

        # Vooruit
        if k == "z":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, x=MOVE_SPEED), loop)

        # Achteruit
        elif k == "s":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, x=-MOVE_SPEED), loop)

        # Draaien links
        elif k == "a":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, z=TURN_SPEED), loop)

        # Draaien rechts
        elif k == "e":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, z=-TURN_SPEED), loop)

        # Zijwaards links
        elif k == "q":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, y=MOVE_SPEED), loop)

        # Zijwaards rechts
        elif k == "d":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, y=-MOVE_SPEED), loop)

        # /
        elif k == " ":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, x=0), loop)

        # Hello
        elif k == "h":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["Hello"]}
                ), loop)

        # Stretch
        elif k == "x":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": 1033, "parameter": {"data": False}}
                ), loop)
        elif k == "y": # sit
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": 1009, "parameter": {"data": False}}
                ), loop)
        elif k == "t": # rize sit
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": 1010, "parameter": {"data": False}}
                ), loop)
        elif k == "f": # Wiggle
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": 1033, "parameter": {"data": False}}
                ), loop)


        # MoonWalk (NON FUNCTIONAL)
        elif k == "m":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["MoonWalk"], "parameter": {"value": 1}}
                ), loop)

        # Neerliggen
        elif k == "l":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["StandDown"]}
                ), loop)

        # Opstaan
        elif k == "o":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["RecoveryStand"], "parameter": {"data": False}}
                ), loop)

        # Zaklamp aan
        elif k == "i":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"], 
                    {"api_id": 1005, "parameter": {"brightness": BRIGHTNESS_LVL}}
                ), loop)

        # White
        elif k == "1":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.WHITE}}
                ), loop)
        # Red
        elif k == "2":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.RED}}
                ), loop)
        # Yellow
        elif k == "3":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.YELLOW}}
                ), loop)
        # Green
        elif k == "4":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.GREEN}}
                ), loop)
        # Cyan
        elif k == "5":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.CYAN}}
                ), loop)
        # Blue
        elif k == "6":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.BLUE}}
                ), loop)
        # Purple
        elif k == "7":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.PURPLE}}
                ), loop)
        # Disco
        elif k == "8":
            for i in range(30):
                select_disco_colour = random.choice(DISCO_COLOURS)

                asyncio.run_coroutine_threadsafe(
                    conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["VUI"],
                        {"api_id": 1007, "parameter": {"color": select_disco_colour}}
                    ), loop)
                time.sleep(0.2)
        # Flash
        elif k == "9":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": 1007, "parameter": {"color": VUI_COLOR.WHITE, "flash_cycle": 500}}
                ), loop)
        # Licht uit
        #elif k == "0":

        # Zaklamp uit
        elif k == "u":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"], 
                    {"api_id": 1005, "parameter": {"brightness": 0}}
                ), loop)

        # EMERGENCY STOP
        elif k == "r":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, x=0, z=0), loop)

        # Print controls
        elif k == "p":
            print_controls()


async def main():
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip="192.168.0.73"
    )

    await conn.connect()

    # Ensure normal mode
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001}
    )
    data = json.loads(response['data']['data'])
    if data["name"] != "normal":
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": "normal"}}
        )
        await asyncio.sleep(5)

    print_controls()

    loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    # Run keyboard listener in a background thread so it doesn't block the async loop
    kb_thread = threading.Thread(
        target=keyboard_listener,
        args=(conn, loop, stop_event),
        daemon=True
    )
    kb_thread.start()

    # Wait until stop_event is set (Ctrl+C in the terminal)
    while not stop_event.is_set():
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted")
        sys.exit(0)
