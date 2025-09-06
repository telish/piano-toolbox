"""Module to send OSC messages to a specified port."""

from pythonosc.udp_client import SimpleUDPClient

_state = {"osc_port": 0, "osc_client": None}


def configure(ip_addr: str, port: int) -> None:
    _state["osc_port"] = port
    _state["osc_client"] = SimpleUDPClient(ip_addr, _state["osc_port"])


def send_message(address: str, *args: object) -> None:
    _state["osc_client"].send_message(address, args)


configure("127.0.0.1", 9876)  # Default IP and port
