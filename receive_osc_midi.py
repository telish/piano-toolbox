import signal
import sys
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


# Function to handle incoming MIDI messages
def handle_midi(address, *args):
    if len(args) == 3:
        channel, note, velocity = args
        print(f"Received MIDI Note: Channel={channel}, Note={note}, Velocity={velocity}")
    else:
        print(f"Unexpected MIDI message: {args}")


# Function to handle SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print("\nShutting down OSC server...")
    sys.exit(0)


# Register the SIGINT handler
signal.signal(signal.SIGINT, signal_handler)

# Set up the OSC dispatcher
dispatcher = Dispatcher()
dispatcher.map("/midi", handle_midi)

# Define and start the OSC server
ip = "0.0.0.0"
port = 8000
server = BlockingOSCUDPServer((ip, port), dispatcher)

print(f"Listening for OSC MIDI messages on {ip}:{port}... (Press Ctrl+C to stop)")
try:
    server.serve_forever()
except KeyboardInterrupt:
    signal_handler(None, None)  # Handle manual KeyboardInterrupt
