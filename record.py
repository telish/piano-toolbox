import subprocess
import signal
import time

# Start all programs
record_video = subprocess.Popen(["python", "record-video.py"])
record_midi = subprocess.Popen(["python", "record-midi.py"])
record_audio = subprocess.Popen(["python", "record-audio.py"])


def stop_processes(sig, frame):
    print("Stopping processes...")
    record_video.send_signal(signal.SIGINT)
    record_midi.send_signal(signal.SIGINT)
    record_audio.send_signal(signal.SIGINT)
    record_video.wait()
    record_midi.wait()
    record_audio.wait()
    print("Processes stopped.")
    exit(0)


# Handle KeyboardInterrupt (Ctrl+C)
signal.signal(signal.SIGINT, stop_processes)

try:
    while True:
        time.sleep(1)  # Keep the script running
except KeyboardInterrupt:
    stop_processes(None, None)
