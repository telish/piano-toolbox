"""Records audio from the default microphone and saves it as a WAV file."""

import wave
import os
import signal
import sys

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000  # 48 kHz
CHANNELS = 1  # Mono
FORMAT_PCM = np.int16  # 16-bit PCM

OUTPUT_DIR = "recording/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = os.path.join(OUTPUT_DIR, "audio.wav")

print(sd.query_devices())  # List all audio devices

# Storage for recorded data
audio_data = []


# Signal handler to stop recording on Ctrl + C
def signal_handler(sig, frame):
    print("\nrecord-audio.py: Stopping")
    save_wav()
    sys.exit(0)


def save_wav():
    if len(audio_data) == 0:
        return

    # Combine all recorded chunks
    audio_np = np.concatenate(audio_data, axis=0)

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


# Attach signal handler for Ctrl + C
signal.signal(signal.SIGINT, signal_handler)

# Start recording
print("record-audio: Recording. Press Ctrl + C to stop.")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=FORMAT_PCM, callback=callback):
    while True:
        sd.sleep(100)  # Keep process running
