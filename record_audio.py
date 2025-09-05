"""Records audio from the default microphone and saves it as a WAV file."""

import wave
import os
import signal
import sys
import argparse

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000  # 48 kHz
CHANNELS = 1  # Mono
FORMAT_PCM = np.int16  # 16-bit PCM

# Storage for recorded data
audio_data = []


# Signal handler to stop recording on Ctrl + C
def signal_handler(sig, frame):
    print("\nrecord-audio.py: Stopping")
    save_wav()
    sys.exit(0)


def save_wav(output_dir="recording/audio"):
    if len(audio_data) == 0:
        return

    # Combine all recorded chunks
    audio_np = np.concatenate(audio_data, axis=0)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "audio.wav")

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_np.tobytes())

    print(f"record-audio.py: Audio saved as {filename}")


# Callback function for non-blocking recording
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_data.append(indata.copy())


def parse_args():
    parser = argparse.ArgumentParser(description="Record audio from microphone")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recording/audio",
        help="Output directory for the WAV file",
    )
    return parser.parse_args()


def main(output_dir=None):
    """
    Start recording audio.

    Args:
        output_dir: Optional output directory for the WAV file
    """
    global audio_data

    # Clear any previous audio data
    audio_data = []

    # Set output directory from argument or command line
    if output_dir is None:
        args = parse_args()
        output_dir = args.output_dir

    print(sd.query_devices())  # List all audio devices

    # Attach signal handler for Ctrl + C
    signal.signal(signal.SIGINT, signal_handler)

    # Start recording
    print("record-audio: Recording. Press Ctrl + C to stop.")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=FORMAT_PCM,
            callback=callback,
        ):
            while True:
                sd.sleep(100)  # Keep process running
    except KeyboardInterrupt:
        save_wav(output_dir)


if __name__ == "__main__":
    main()
