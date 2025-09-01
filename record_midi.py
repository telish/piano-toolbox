import os
import time

import mido


midi_port_name = None

# Automatically select the first available MIDI input port
available_ports = mido.get_input_names()
if available_ports:
    midi_port_name = available_ports[0]
    print(f"record-midi.py: Using MIDI input: {midi_port_name}")
else:
    print("record-midi.py: No MIDI input ports available.")
    exit()

output_dir = "recording/midi"
os.makedirs(output_dir, exist_ok=True)

# Open MIDI input
try:
    with mido.open_input(midi_port_name) as inport:
        print("record-midi.py: Recording. Press Ctrl+C to stop.")
        filename = os.path.join(output_dir, "midi_msg.txt")
        with open(filename, "w") as file:
            while True:
                for msg in inport.iter_pending():
                    timestamp = time.time()
                    if msg.type in ["note_off", "note_on", "control_change"]:
                        log_entry = f"{timestamp:.7f}: {msg}\n"
                        file.write(log_entry)
                        print(log_entry.strip())
except KeyboardInterrupt:
    print("record-midi.py: Stopped.")
