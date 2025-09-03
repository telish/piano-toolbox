from pythonosc.udp_client import SimpleUDPClient

osc_port: int
osc_client: SimpleUDPClient


def configure(port: int):
    global osc_port, osc_client
    osc_port = port
    osc_client = SimpleUDPClient("127.0.0.1", osc_port)


def send_message(address: str, *args):
    global osc_client
    osc_client.send_message(address, args)
