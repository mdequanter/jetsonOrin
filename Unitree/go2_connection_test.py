"""
Go2 WebRTC verbindings- en diagnosetest
========================================

Go2-variant van g1_connection_test.py. Zelfde aanpak, maar afgestemd op de
Go2 (Pro): de Go2 gebruikt het SPORT_MOD-topic en SPORT_CMD["Move"], en we
zetten de robot eerst in 'normal' mode (net als keyboard2.py doet).

Dit script GOKT niets. Het:
  1) maakt verbinding,
  2) drukt af welke RTC_TOPIC / commando-tabellen de library kent,
  3) zet motion-switcher op 'normal' en toont de status,
  4) doet -- alleen met --move en na bevestiging -- één klein, kort
     loopcommando en stopt meteen weer.

Gebruik:
    # Automatisch de Go2 op het netwerk zoeken (geen IP nodig):
    python go2_connection_test.py

    # Of expliciet op IP / serienummer:
    python go2_connection_test.py --ip 192.168.0.73 [--aes-key <32-hex>]
    python go2_connection_test.py --serial B42D2000XXXXXXXX

    # Interactief acties uit een tabel (SPORT_CMD, ...) kiezen en uitvoeren:
    python go2_connection_test.py --ip 192.168.0.73 --actions

    # Of één klein vooruit-commando (Go2 moet vrij staan):
    python go2_connection_test.py --ip 192.168.0.73 --move

Op de meeste Go2-firmware is GEEN AES-sleutel nodig (--aes-key weglaten).
"""

import argparse
import asyncio
import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor

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


# Poorten waarop de Go2 zijn WebRTC-signaling aanbiedt (zie de foutmeldingen:
# /offer op 8081 = oude methode, /con_notify op 9991 = nieuwe methode).
SIGNALING_PORTS = (8081, 9991)


def _local_ipv4():
    """Bepaal het primaire IPv4-adres van deze machine (zonder pakketten te sturen)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # verbindt niet echt; kiest wel de juiste interface
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _probe(ip, timeout=0.5):
    """True als 'ip' een TCP-verbinding op een van de signaling-poorten accepteert."""
    for port in SIGNALING_PORTS:
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def discover_ip():
    """Scan het lokale /24-subnet naar een host die op de Go2-signaling-poorten luistert."""
    local = _local_ipv4()
    if not local:
        print("Kon lokaal IP niet bepalen -- discovery afgebroken.")
        return None
    prefix = local.rsplit(".", 1)[0]
    print(f"Scannen van {prefix}.0/24 op poorten {SIGNALING_PORTS} "
          f"(vanaf {local}, duurt enkele seconden) ...")
    candidates = [f"{prefix}.{i}" for i in range(1, 255)
                  if f"{prefix}.{i}" != local]
    found = []
    with ThreadPoolExecutor(max_workers=128) as ex:
        for ip, ok in zip(candidates, ex.map(_probe, candidates)):
            if ok:
                found.append(ip)
    if not found:
        print("Geen robot gevonden op de signaling-poorten in dit subnet.")
        return None
    if len(found) > 1:
        print(f"Meerdere kandidaten gevonden: {found} -- de eerste wordt gebruikt.")
    print(f"Robot gevonden op: {found[0]}")
    return found[0]


def dump_constants():
    """Druk alle relevante tabellen uit de constants-module af."""
    print("\n================ BESCHIKBARE CONSTANTS ================")
    for name in dir(C):
        if name.startswith("_"):
            continue
        value = getattr(C, name)
        if isinstance(value, dict):
            print(f"\n[{name}]  (dict, {len(value)} entries)")
            for k, v in value.items():
                print(f"    {k!r:32} -> {v!r}")
    print("======================================================\n")


async def query(conn, topic_key, payload, label):
    """Verstuur één request en druk het antwoord af."""
    topics = getattr(C, "RTC_TOPIC", {})
    if topic_key not in topics:
        print(f"  [overslaan] topic {topic_key!r} bestaat niet in deze versie.")
        return None
    print(f"  -> {label}  (topic={topic_key}, payload={payload})")
    try:
        resp = await conn.datachannel.pub_sub.publish_request_new(
            topics[topic_key], payload
        )
        print(f"     antwoord: {resp}")
        return resp
    except Exception as e:  # noqa: BLE001
        print(f"     FOUT bij {label}: {e!r}")
        return None


async def ensure_normal_mode(conn):
    """Zet de Go2 in 'normal' sport-mode (zoals keyboard2.py)."""
    resp = await query(conn, "MOTION_SWITCHER", {"api_id": 1001}, "get mode")
    if not resp:
        return
    try:
        data = json.loads(resp["data"]["data"])
        current = data.get("name")
        print(f"     huidige mode: {current!r}")
        if current != "normal":
            print("     -> omschakelen naar 'normal' ...")
            await query(
                conn, "MOTION_SWITCHER",
                {"api_id": 1002, "parameter": {"name": "normal"}},
                "set normal",
            )
            await asyncio.sleep(5)
    except Exception as e:  # noqa: BLE001
        print(f"     kon mode niet parsen: {e!r}")


async def ask(prompt):
    """Vraag invoer zonder de asyncio-loop (en dus de WebRTC-verbinding) te blokkeren."""
    return (await asyncio.to_thread(input, prompt)).strip()


def command_tables():
    """Alle niet-lege dicts uit de constants-module (SPORT_CMD, RTC_TOPIC, ...)."""
    tables = {}
    for name in dir(C):
        if name.startswith("_"):
            continue
        val = getattr(C, name)
        if isinstance(val, dict) and val:
            tables[name] = val
    return tables


async def action_runner(conn):
    """Interactief acties uit een tabel (bv. SPORT_CMD) kiezen en uitvoeren.

    Zo kun je op de Go2 uitproberen welke acties werken -- en later exact
    hetzelfde doen op de G1 om te zien wat daar verschilt.
    """
    tables = command_tables()
    topics = getattr(C, "RTC_TOPIC", {})
    if not tables:
        print("Geen commandotabellen gevonden in constants.")
        return

    # Standaard-topic: SPORT_MOD als het bestaat (Go2), anders de eerste.
    default_topic = "SPORT_MOD" if "SPORT_MOD" in topics else (
        next(iter(topics), None))

    while True:
        table_names = list(tables.keys())
        print("\n===== TABELLEN =====")
        for i, name in enumerate(table_names):
            print(f"  {i:2}) {name}  ({len(tables[name])} entries)")
        pick = await ask("Kies tabel (nummer), of 'q' om te stoppen: ")
        if pick.lower() in ("q", "quit", "exit"):
            return
        if not pick.isdigit() or int(pick) >= len(table_names):
            print("Ongeldige keuze.")
            continue
        table_name = table_names[int(pick)]
        table = tables[table_name]

        # Acties binnen de gekozen tabel.
        while True:
            entries = list(table.items())
            print(f"\n----- {table_name} -----")
            for i, (k, v) in enumerate(entries):
                print(f"  {i:2}) {k:28} -> {v!r}")
            sel = await ask("Kies actie (nummer of naam), 'b' terug, 'q' stop: ")
            if sel.lower() in ("q", "quit", "exit"):
                return
            if sel.lower() in ("b", "back"):
                break

            # Resolven op index of op naam.
            key = api_id = None
            if sel.isdigit() and int(sel) < len(entries):
                key, api_id = entries[int(sel)]
            elif sel in table:
                key, api_id = sel, table[sel]
            else:
                print("Ongeldige keuze.")
                continue

            if not isinstance(api_id, int):
                print(f"Waarde van {key!r} is geen api_id (int): {api_id!r} -- overslaan.")
                continue

            # Doel-topic (met verstandige default).
            topic_in = await ask(f"Topic [{default_topic}]: ")
            topic_key = topic_in or default_topic
            if topic_key not in topics:
                print(f"Topic {topic_key!r} bestaat niet. Beschikbaar: {list(topics)}")
                continue

            # Optionele parameter als JSON (bv. {\"x\": 0.2} voor Move).
            param_in = await ask('Parameter als JSON (leeg = geen), bv {"x": 0.3, "y": 0, "z": 0} :')
            payload = {"api_id": api_id}
            if param_in:
                try:
                    payload["parameter"] = json.loads(param_in)
                except json.JSONDecodeError as e:
                    print(f"Ongeldige JSON: {e} -- actie geannuleerd.")
                    continue

            confirm = await ask(
                f"!!! '{key}' sturen naar {topic_key}. Dit kan de robot laten "
                f"BEWEGEN. Typ 'ja': ")
            if confirm.lower() != "ja":
                print("Geannuleerd.")
                continue

            print(f"  -> versturen: {payload}")
            try:
                resp = await conn.datachannel.pub_sub.publish_request_new(
                    topics[topic_key], payload)
                print(f"     antwoord: {resp}")
            except Exception as e:  # noqa: BLE001
                print(f"     FOUT: {e!r}")


async def move_test(conn):
    """Eén klein, kort loopcommando voor de Go2 -- alleen na bevestiging."""
    topics = getattr(C, "RTC_TOPIC", {})
    sport_cmd = getattr(C, "SPORT_CMD", {})
    if "SPORT_MOD" not in topics or "Move" not in sport_cmd:
        print("SPORT_MOD-topic of Move-commando ontbreekt -- test afgebroken.")
        return

    print("\n!!! De Go2 gaat zo héél even proberen vooruit te bewegen. !!!")
    answer = input("Staat de Go2 rechtop, in vrije ruimte? Typ 'ja' om door te gaan: ")
    if answer.strip().lower() != "ja":
        print("Geannuleerd. Geen beweging verzonden.")
        return

    move_id = sport_cmd["Move"]
    try:
        print("   -> klein vooruit-commando (x=0.2) ...")
        await conn.datachannel.pub_sub.publish_request_new(
            topics["SPORT_MOD"],
            {"api_id": move_id, "parameter": {"x": 0.2, "y": 0.0, "z": 0.0}},
        )
        await asyncio.sleep(0.8)
    finally:
        print("   -> STOP (x=0) ...")
        await conn.datachannel.pub_sub.publish_request_new(
            topics["SPORT_MOD"],
            {"api_id": move_id, "parameter": {"x": 0.0, "y": 0.0, "z": 0.0}},
        )
    print("Bewegingstest klaar.")


def report_ip(conn):
    """Best-effort: druk het gevonden IP af als de library het beschikbaar stelt."""
    for attr in ("ip", "robot_ip", "host", "_ip"):
        val = getattr(conn, attr, None)
        if val:
            print(f"Gevonden robot-IP: {val}")
            return


async def main(args):
    # Bouw de verbindingsargumenten op op basis van wat is opgegeven:
    #   --ip     -> rechtstreeks op dat IP
    #   --serial -> multicast-discovery op serienummer
    #   niets    -> automatische multicast-discovery op het LAN (library zoekt zelf)
    kwargs = {}
    if args.ip:
        kwargs["ip"] = args.ip
        where = f"IP {args.ip}"
    elif args.serial:
        kwargs["serialNumber"] = args.serial
        where = f"serienummer {args.serial} (multicast-discovery)"
    else:
        print("Geen IP/serienummer opgegeven -> zoeken naar de Go2 op het LAN ...")
        found_ip = await asyncio.to_thread(discover_ip)
        if not found_ip:
            print("\nKon de robot niet automatisch vinden.")
            print("Geef expliciet --ip <adres> of --serial <serienummer> op,")
            print("en controleer dat je op hetzelfde netwerk zit als de Go2.")
            return
        kwargs["ip"] = found_ip
        where = f"gevonden IP {found_ip}"
    if args.aes_key:
        kwargs["aes_128_key"] = args.aes_key

    method = getattr(WebRTCConnectionMethod, args.method)
    print(f"Verbinden met Go2 via {args.method} -- {where} ...")
    conn = UnitreeWebRTCConnection(method, **kwargs)

    try:
        await conn.connect()
    except Exception as e:  # noqa: BLE001
        print(f"\nVERBINDING MISLUKT: {e!r}")
        print("Controleer: juist IP/serienummer? zelfde netwerk? AES-sleutel nodig?")
        return
    print("Verbinding OK.")
    report_ip(conn)
    print()

    # 1) Welke topics/commando's kent deze versie?
    dump_constants()

    # 2) Sport-mode op 'normal' + status.
    print("Motion-switcher:")
    await ensure_normal_mode(conn)

    # 3) Optioneel: interactieve actie-runner of enkele bewegingstest.
    if args.actions:
        await action_runner(conn)
    elif args.move:
        await move_test(conn)
    else:
        print("\n(Geen actie uitgevoerd. Voeg --actions toe om acties uit een")
        print(" tabel zoals SPORT_CMD interactief te kiezen en uit te voeren,")
        print(" of --move voor één klein vooruit-commando.)")

    print("\nKlaar. Verbinding wordt afgesloten.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 WebRTC verbindings-/diagnosetest")
    parser.add_argument("--ip", default=None,
                        help="IP-adres van de Go2. Weglaten = automatisch zoeken op het LAN")
    parser.add_argument("--serial", default=None,
                        help="Serienummer voor multicast-discovery (bv. B42D2000XXXXXXXX)")
    parser.add_argument("--aes-key", default=None,
                        help="AES-128-sleutel (meestal niet nodig bij Go2)")
    parser.add_argument("--method", default="LocalSTA",
                        help="Verbindingsmethode (LocalSTA / LocalAP / Remote)")
    parser.add_argument("--move", action="store_true",
                        help="Voer na bevestiging één klein loopcommando uit")
    parser.add_argument("--actions", action="store_true",
                        help="Interactief acties uit een tabel (bv. SPORT_CMD) kiezen en uitvoeren")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nOnderbroken.")
        sys.exit(0)
