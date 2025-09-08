"""Record MIDI messages from a specified input port and save them to a text file."""

import argparse
import os
import sys
import time

import mido

parser = argparse.ArgumentParser(description="Record MIDI messages from input port.")
parser.add_argument(
    "--port-index",
    type=int,
    default=0,
    help="Index of the MIDI input port to use (default: 0)",
)
args = parser.parse_args()

# List available MIDI input ports
available_ports = mido.get_input_names()
if not available_ports:
    print("record-midi.py: No MIDI input ports found.")
    sys.exit(1)

# Select MIDI port by index
if args.port_index < 0 or args.port_index >= len(available_ports):
    print(f"record-midi.py: Invalid port index {args.port_index}. Available ports:")
    for i, port in enumerate(available_ports):
        print(f"  [{i}] {port}")
    sys.exit(1)

midi_port_name = available_ports[args.port_index]
print(f"record-midi.py: Using MIDI input: {midi_port_name}")

OUTPUT_DIR = "recording/midi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open MIDI input
try:
    with mido.open_input(midi_port_name) as inport:  # type: ignore
        print("record-midi.py: Recording. Press Ctrl+C to stop.")
        filename = os.path.join(OUTPUT_DIR, "midi_msg.txt")
        with open(filename, "w", encoding="utf-8") as file:
            while True:
                for msg in inport.iter_pending():
                    timestamp = time.time()
                    if msg.type in ["note_off", "note_on", "control_change"]:
                        log_entry = f"{timestamp:.7f}: {msg}\n"
                        file.write(log_entry)
                        print(log_entry.strip())
except KeyboardInterrupt:
    print("record-midi.py: Stopped.")
