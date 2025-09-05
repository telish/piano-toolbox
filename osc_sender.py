"""Module to send OSC messages to a specified port."""

from pythonosc.udp_client import SimpleUDPClient

_state = {"osc_port": 0, "osc_client": None}


def configure(port: int):
    _state["osc_port"] = port
    _state["osc_client"] = SimpleUDPClient("127.0.0.1", _state["osc_port"])


def send_message(address: str, *args):
    _state["osc_client"].send_message(address, args)


configure(9876)  # Default port
