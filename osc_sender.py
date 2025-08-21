from pythonosc.udp_client import SimpleUDPClient

osc_port = None
osc_client = None


def configure(port):
    global osc_port, osc_client
    osc_port = port
    osc_client = SimpleUDPClient("127.0.0.1", osc_port)


def send_message(address, *args):
    global osc_client
    assert osc_client is not None, "OSC client is not configured. Call configure() first."
    if osc_client is not None:
        osc_client.send_message(address, args)
