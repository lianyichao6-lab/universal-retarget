"""Manual Pico 4 discovery/relay daemon.

Start this script when you want Pico 4 / VR to discover the current PC
without starting a business script first. It:

1. broadcasts the PC IP on UDP so Pico 4 can discover it
2. accepts Pico 4 direct TCP connection
3. republishes tracking JSON to local relay clients on 127.0.0.1:63902

Application scripts can then keep using Pico4 relay mode.
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import threading
import time

from pico4 import (
    _CMD_BATTERY,
    _CMD_CONNECT,
    _CMD_DEVICE_STATE_JSON,
    _CMD_HEARTBEAT,
    _CMD_SENSOR,
    _DirectFrameParser,
    _build_broadcast_packet,
    _get_local_ips,
)

logger = logging.getLogger("pico4_daemon")

DEFAULT_DIRECT_PORT = 63901
DEFAULT_RELAY_HOST = "127.0.0.1"
DEFAULT_RELAY_PORT = 63902
DEFAULT_BROADCAST_PORT = 29888
DEFAULT_DEVICE_ID = "pico4"
HEARTBEAT_TIMEOUT_S = 20.0
BROADCAST_INTERVAL_S = 5.0


def encode_relay_frame(device_id: str, payload: bytes) -> bytes:
    device_id_bytes = device_id.encode("utf-8")
    return (
        struct.pack("<I", len(device_id_bytes))
        + device_id_bytes
        + struct.pack("<I", len(payload))
        + payload
    )


class RelayHub:
    def __init__(self, host: str, port: int, device_id: str) -> None:
        self._host = host
        self._port = port
        self._device_id = device_id
        self._stop = threading.Event()
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for client in list(self._clients):
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()
        self._thread.join(timeout=2.0)

    def publish(self, payload: bytes) -> None:
        frame = encode_relay_frame(self._device_id, payload)
        dead_clients: list[socket.socket] = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(frame)
                except OSError:
                    dead_clients.append(client)
            for client in dead_clients:
                self._clients.discard(client)
                try:
                    client.close()
                except OSError:
                    pass

    def _run(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(8)
        server.settimeout(1.0)
        logger.info("Relay hub listening on %s:%d", self._host, self._port)
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                logger.info("Relay client connected: %s", addr)
                with self._lock:
                    self._clients.add(conn)
        finally:
            server.close()


class Pico4Daemon:
    def __init__(
        self,
        direct_port: int,
        relay_host: str,
        relay_port: int,
        broadcast_port: int,
        device_id: str,
    ) -> None:
        self._direct_port = direct_port
        self._broadcast_port = broadcast_port
        self._hub = RelayHub(relay_host, relay_port, device_id)
        self._stop = threading.Event()

    def run(self) -> None:
        self._hub.start()
        broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        broadcast_thread.start()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self._direct_port))
        server.listen(1)
        server.settimeout(1.0)
        logger.info("Direct server listening on 0.0.0.0:%d", self._direct_port)

        try:
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                logger.info("Pico 4 connected from %s", addr)
                self._handle_direct_client(conn)
                logger.info("Pico 4 disconnected")
        except KeyboardInterrupt:
            logger.info("Stopping daemon...")
        finally:
            self._stop.set()
            server.close()
            self._hub.stop()

    def _handle_direct_client(self, conn: socket.socket) -> None:
        parser = _DirectFrameParser()
        conn.settimeout(1.0)
        last_heartbeat = time.monotonic()
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_heartbeat > HEARTBEAT_TIMEOUT_S:
                    logger.warning("Pico 4 heartbeat timeout")
                    break
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                parser.feed(data)
                while True:
                    frame = parser.try_parse()
                    if frame is None:
                        break
                    if frame["cmd"] in (
                        _CMD_HEARTBEAT,
                        _CMD_CONNECT,
                        _CMD_BATTERY,
                        _CMD_SENSOR,
                    ):
                        last_heartbeat = time.monotonic()
                    if frame["cmd"] == _CMD_DEVICE_STATE_JSON:
                        self._hub.publish(frame["payload"])
        finally:
            conn.close()

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                for ip in _get_local_ips():
                    parts = ip.split(".")
                    if len(parts) != 4:
                        continue
                    broadcast_ip = ".".join(parts[:3]) + ".255"
                    packet = _build_broadcast_packet(ip)
                    try:
                        sock.sendto(packet, (broadcast_ip, self._broadcast_port))
                    except OSError:
                        pass
                self._stop.wait(BROADCAST_INTERVAL_S)
        finally:
            sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Pico 4 discovery/relay daemon")
    parser.add_argument("--direct-port", type=int, default=DEFAULT_DIRECT_PORT)
    parser.add_argument("--relay-host", default=DEFAULT_RELAY_HOST)
    parser.add_argument("--relay-port", type=int, default=DEFAULT_RELAY_PORT)
    parser.add_argument("--broadcast-port", type=int, default=DEFAULT_BROADCAST_PORT)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    daemon = Pico4Daemon(
        direct_port=args.direct_port,
        relay_host=args.relay_host,
        relay_port=args.relay_port,
        broadcast_port=args.broadcast_port,
        device_id=args.device_id,
    )
    daemon.run()


if __name__ == "__main__":
    main()
