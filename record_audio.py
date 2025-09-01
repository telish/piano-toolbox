import sounddevice as sd
import wave
import numpy as np
import os
import signal
import sys

samplerate = 48000  # 48 kHz
channels = 1  # Mono
dtype = np.int16  # 16-bit PCM

output_dir = "recording/audio"
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, "audio.wav")

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
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
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
with sd.InputStream(
    samplerate=samplerate, channels=channels, dtype=dtype, callback=callback
):
    while True:
        sd.sleep(100)  # Keep process running
