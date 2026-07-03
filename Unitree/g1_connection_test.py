"""
G1 WebRTC verbindings- en diagnosetest
======================================

Doel: op de Jetson/laptop (mét Rob-e in hetzelfde netwerk) uitzoeken of de
consumenten-G1 (Basic) bereikbaar is via dezelfde WebRTC-route als de app,
en welke topics/commando's jouw versie van `unitree_webrtc_connect` voor de
G1 aanbiedt.

Dit script GOKT geen api_id's. Het:
  1) maakt verbinding,
  2) drukt af welke RTC_TOPIC / commando-tabellen de library kent
     (zodat we de juiste G1-loco-id's zien i.p.v. te raden),
  3) vraagt de motion_switcher-status op (bewijst dat het datakanaal werkt),
  4) doet -- alleen met de vlag --move en na bevestiging -- één klein,
     kortstondig loopcommando en stopt meteen weer.

Gebruik:
    pip install unitree_webrtc_connect
    unitree-fetch-aes-key            # firmware >= 1.5.1: haalt AES-sleutel op

    # Alleen verbinden + info afdrukken (VEILIG, robot beweegt niet):
    python g1_connection_test.py --ip 192.168.x.x [--aes-key <32-hex>]

    # Pas als 1-3 werken en de robot vrij staat, met bewegingstest:
    python g1_connection_test.py --ip 192.168.x.x --aes-key <32-hex> --move

Let op: zorg dat Rob-e via de app RECHTOP staat, in een open ruimte, en dat
iemand klaarstaat om op te vangen voordat je --move gebruikt.
"""

import argparse
import asyncio
import json
import sys

try:
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )
    from unitree_webrtc_connect import constants as C
except ImportError as e:
    print("FOUT: kon 'unitree_webrtc_connect' niet importeren.")
    print("      Installeer het eerst:  pip install unitree_webrtc_connect")
    print(f"      Detail: {e}")
    sys.exit(1)


def dump_constants():
    """Druk alle relevante tabellen uit de constants-module af.

    Zo zien we exact welke topics en commando-id's DEZE versie voor de G1
    kent -- daaruit halen we straks de echte loco-Move-id i.p.v. te gokken.
    """
    print("\n================ BESCHIKBARE CONSTANTS ================")
    for name in dir(C):
        if name.startswith("_"):
            continue
        value = getattr(C, name)
        # We tonen alleen dicts/tabellen en enum-achtige objecten.
        if isinstance(value, dict):
            print(f"\n[{name}]  (dict, {len(value)} entries)")
            for k, v in value.items():
                print(f"    {k!r:32} -> {v!r}")
    print("\n(Zoek hierboven naar iets als LOCO / loco / SPORT_MOD en de")
    print(" bijhorende 'Move'-achtige commando's -- die hebben we nodig")
    print(" voor keyboardG1.py.)")
    print("======================================================\n")


def find_topic(*candidates):
    """Zoek de eerste bestaande RTC_TOPIC-sleutel uit een lijst kandidaten."""
    topics = getattr(C, "RTC_TOPIC", {})
    for key in candidates:
        if key in topics:
            return key, topics[key]
    return None, None


async def query(conn, topic_key, api_id, label):
    """Verstuur één request en druk het antwoord af. Retourneert de response."""
    topics = getattr(C, "RTC_TOPIC", {})
    if topic_key not in topics:
        print(f"  [overslaan] topic {topic_key!r} bestaat niet in deze versie.")
        return None
    print(f"  -> {label}  (topic={topic_key}, api_id={api_id})")
    try:
        resp = await conn.datachannel.pub_sub.publish_request_new(
            topics[topic_key], {"api_id": api_id}
        )
        print(f"     antwoord: {resp}")
        return resp
    except Exception as e:  # noqa: BLE001 - diagnose, we willen alles zien
        print(f"     FOUT bij {label}: {e!r}")
        return None


async def move_test(conn):
    """Eén klein, kort loopcommando -- alleen na expliciete bevestiging.

    We proberen de loco-topic te vinden. De exacte 'Move'-api_id verschilt
    per firmware/library-versie; we proberen een aantal bekende sleutels en
    stoppen hoe dan ook direct weer met snelheid 0.
    """
    topic_key, topic_val = find_topic("LOCO", "SPORT_MOD")
    if not topic_key:
        print("Geen loco/sport-topic gevonden -- bewegingstest afgebroken.")
        print("Bekijk de constants-dump hierboven voor de juiste topicnaam.")
        return

    print(f"\nBewegingstopic gevonden: {topic_key} -> {topic_val}")
    print("!!! De robot gaat zo héél even proberen vooruit te bewegen. !!!")
    answer = input("Staat Rob-e rechtop, in vrije ruimte? Typ 'ja' om door te gaan: ")
    if answer.strip().lower() != "ja":
        print("Geannuleerd. Geen beweging verzonden.")
        return

    # Zoek een 'Move'-commando-id. We proberen SPORT_CMD['Move'] (Go2-stijl)
    # en anders een generieke waarde die je na de constants-dump kunt bijstellen.
    move_id = None
    sport_cmd = getattr(C, "SPORT_CMD", {})
    if "Move" in sport_cmd:
        move_id = sport_cmd["Move"]
    if move_id is None:
        print("Geen 'Move'-id in SPORT_CMD gevonden.")
        print("Vul de juiste loco-Move-id in (zie constants-dump) en herstart.")
        return

    topics = getattr(C, "RTC_TOPIC", {})
    try:
        print("   -> klein vooruit-commando (vx=0.1) ...")
        await conn.datachannel.pub_sub.publish_request_new(
            topics[topic_key],
            {"api_id": move_id, "parameter": {"x": 0.1, "y": 0.0, "z": 0.0}},
        )
        await asyncio.sleep(0.8)
    finally:
        # Wat er ook gebeurt: STOP.
        print("   -> STOP (vx=0) ...")
        await conn.datachannel.pub_sub.publish_request_new(
            topics[topic_key],
            {"api_id": move_id, "parameter": {"x": 0.0, "y": 0.0, "z": 0.0}},
        )
    print("Bewegingstest klaar.")


async def main(args):
    # Bouw de verbindingsargumenten op; AES-sleutel alleen meegeven indien opgegeven.
    kwargs = {"ip": args.ip}
    if args.aes_key:
        kwargs["aes_128_key"] = args.aes_key

    method = getattr(WebRTCConnectionMethod, args.method)
    print(f"Verbinden met G1 op {args.ip} via {args.method} ...")
    conn = UnitreeWebRTCConnection(method, **kwargs)

    try:
        await conn.connect()
    except Exception as e:  # noqa: BLE001
        print(f"\nVERBINDING MISLUKT: {e!r}")
        print("Controleer: juist IP? zelfde netwerk? AES-sleutel nodig/juist?")
        return
    print("Verbinding OK.\n")

    # 1) Welke topics/commando's kent deze versie?
    dump_constants()

    # 2) Motion switcher status -- bewijst dat het RPC-datakanaal werkt.
    print("Motion-switcher status opvragen (api_id=1001):")
    resp = await query(conn, "MOTION_SWITCHER", 1001, "get mode")
    if resp:
        try:
            data = json.loads(resp["data"]["data"])
            print(f"     huidige mode: {data.get('name')!r}")
        except Exception:  # noqa: BLE001
            pass

    # 3) Optioneel: bewegingstest.
    if args.move:
        await move_test(conn)
    else:
        print("\n(Geen bewegingstest uitgevoerd. Voeg --move toe wanneer de robot")
        print(" rechtop en vrij staat om één klein loopcommando te proberen.)")

    print("\nKlaar. Verbinding wordt afgesloten.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 WebRTC verbindings-/diagnosetest")
    parser.add_argument("--ip", required=True, help="IP-adres van de G1 (STA-mode)")
    parser.add_argument("--aes-key", default=None,
                        help="Per-device AES-128-sleutel (firmware >= 1.5.1)")
    parser.add_argument("--method", default="LocalSTA",
                        help="Verbindingsmethode (LocalSTA / LocalAP / Remote)")
    parser.add_argument("--move", action="store_true",
                        help="Voer na bevestiging één klein loopcommando uit")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nOnderbroken.")
        sys.exit(0)
