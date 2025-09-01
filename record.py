import os
import shutil
import subprocess
import signal
import time
import argparse  # Import argparse for command-line arguments
import shlex  # For splitting argument strings safely

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Start recording processes with arguments."
)
parser.add_argument(
    "--args-video", type=str, default="", help="Arguments for record_video.py"
)
parser.add_argument(
    "--args-audio", type=str, default="", help="Arguments for record_audio.py"
)
parser.add_argument(
    "--args-midi", type=str, default="", help="Arguments for record_midi.py"
)
args = parser.parse_args()

# Delete 'recording' directory if it exists
if os.path.exists("recording"):
    shutil.rmtree("recording")

# Start all programs with forwarded arguments
video_cmd = ["python", "record_video.py"] + shlex.split(args.args_video)
audio_cmd = ["python", "record_audio.py"] + shlex.split(args.args_audio)
midi_cmd = ["python", "record_midi.py"] + shlex.split(args.args_midi)

record_video = subprocess.Popen(video_cmd)
record_midi = subprocess.Popen(midi_cmd)
record_audio = subprocess.Popen(audio_cmd)


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
