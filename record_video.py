import argparse
import json
import os
import signal
import sys
import time

import cv2


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Record video from webcam.")
parser.add_argument(
    "--camera-index",
    type=int,
    default=0,
    help="Index of the camera to use (default: 0)",
)
parser.add_argument(
    "--show-image",
    action="store_true",
    help="Show the video frames during recording",
)
args = parser.parse_args()

# Use the camera index and show_image from command-line arguments
camera_index = args.camera_index
show_image = args.show_image

# Create output directory
output_dir = "recording/video"
os.makedirs(output_dir, exist_ok=True)

# Configure video capture
fourcc = cv2.VideoWriter_fourcc(*"HFYU")  # HuffYUV Codec (lossless)
fps = 30.0
video_filename = os.path.join(output_dir, "recording.avi")
timestamps_filename = os.path.join(output_dir, "timestamps.json")

# Open camera
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("record-video.py: Error. Could not open webcam.")
    sys.exit(1)

# Init video writer
ret, frame = cap.read()
if not ret:
    print("record-video.py: Error. Failed to capture first frame.")
    sys.exit(1)

height, width = frame.shape[:2]
out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
timestamps = []


# Signal handler for graceful exit
def signal_handler(sig, frame):
    # Close video writer and camera
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    # Save timestamps to file
    with open(timestamps_filename, "w") as f:
        json.dump(timestamps, f)

    print(f"record-video.py: Video saved to {video_filename}")
    print(f"record-video.py: Timestamps saved to {timestamps_filename}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

print("record-video.py: Recording camera. Press Ctrl+C to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("record-video.py: Error. Failed to capture frame.")
        break

    # Save frame and timestamp
    out.write(frame)
    timestamps.append({"timestamp": time.time(), "frame_number": len(timestamps)})

    # Show the image in a window if requested
    if show_image:
        cv2.imshow("Recording", frame)

    # Exit recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        signal_handler(signal.SIGINT, None)

# Cleanup (in case loop exits normally)
out.release()
cap.release()
cv2.destroyAllWindows()
