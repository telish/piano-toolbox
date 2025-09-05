"""Start recording video, audio, and MIDI processes with forwarded arguments."""

import argparse  # Import argparse for command-line arguments
import os
import shlex  # For splitting argument strings safely
import shutil
import signal
import subprocess
import time


def parse_args() -> argparse.Namespace:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Start recording processes with arguments.")
    parser.add_argument("--args-video", type=str, default="", help="Arguments for record_video.py")
    parser.add_argument("--args-audio", type=str, default="", help="Arguments for record_audio.py")
    parser.add_argument("--args-midi", type=str, default="", help="Arguments for record_midi.py")
    return parser.parse_args()


def main(args_video: str = "", args_audio: str = "", args_midi: str = "") -> None:
    """
    Start recording processes.

    Args:
        args_video: Optional arguments for record_video.py
        args_audio: Optional arguments for record_audio.py
        args_midi: Optional arguments for record_midi.py
    """
    # Use provided arguments or parse from command line
    if not all([isinstance(arg, str) for arg in [args_video, args_audio, args_midi]]):
        args = parse_args()
        args_video = args.args_video if not isinstance(args_video, str) else args_video
        args_audio = args.args_audio if not isinstance(args_audio, str) else args_audio
        args_midi = args.args_midi if not isinstance(args_midi, str) else args_midi

    # Delete 'recording' directory if it exists and no_clear is False
    if os.path.exists("recording"):
        shutil.rmtree("recording")

    # Start all programs with forwarded arguments
    video_cmd = ["python", "record_video.py"] + shlex.split(args_video)
    audio_cmd = ["python", "record_audio.py"] + shlex.split(args_audio)
    midi_cmd = ["python", "record_midi.py"] + shlex.split(args_midi)

    record_video = subprocess.Popen(video_cmd)
    record_midi = subprocess.Popen(midi_cmd)
    record_audio = subprocess.Popen(audio_cmd)

    def stop_processes(_sig: int, _frame: object) -> None:
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
        stop_processes(0, None)


if __name__ == "__main__":
    main()
