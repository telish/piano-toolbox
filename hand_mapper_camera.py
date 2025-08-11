import signal
import sys

import json
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

with open('config.json', 'r') as file:
    config = json.load(file)
config = config['hand_mapper_camera.py']

# Function to handle incoming MIDI messages
def handle_osc(address, *args):
    if len(args) == 3:
        channel, note, velocity = args
        print(
            f"hand_mapper_camera.py: Channel={channel}, Note={note}, Velocity={velocity}")
    else:
        print(f"hand_mapper_camera.py: Unexpected MIDI message: {args}")


def handle_hands(address, *args):
    print(f"hand_mapper_camera.py: Received OSC message: {address} {args}")


# Function to handle SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print("\nShutting down OSC server...")
    sys.exit(0)


# Register the SIGINT handler
signal.signal(signal.SIGINT, signal_handler)

# Set up the OSC dispatcher
dispatcher = Dispatcher()
dispatcher.map("/midi", handle_osc)
dispatcher.map("/left/*", handle_hands)
dispatcher.map("/right/*", handle_hands)

# Define and start the OSC server
ip = "0.0.0.0"
port = config["port_incoming_hands"]
server = BlockingOSCUDPServer((ip, port), dispatcher)

print(
    f"hand_mapper.camera.py: Listening for OSC MIDI messages on {ip}:{port}. Press Ctrl+C to stop")
try:
    server.serve_forever()
except KeyboardInterrupt:
    signal_handler(None, None)  # Handle manual KeyboardInterrupt
