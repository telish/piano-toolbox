import cv2
import signal
import sys
import os
import time
import json
import utils


config = utils.get_config_for_file(__file__)

# Create output directory
output_dir = "recording/video"
os.makedirs(output_dir, exist_ok=True)

# Configure video capture
fourcc = cv2.VideoWriter_fourcc(*"HFYU")  # HuffYUV Codec (lossless)
fps = 30.0
video_filename = os.path.join(output_dir, "recording.avi")
timestamps_filename = os.path.join(output_dir, "timestamps.json")

# Open camera
cap = cv2.VideoCapture(config.get("camera_index", 0))  # Default to camera index 0
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
    # Close video writer
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    # Save timestamps
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

    # Show the image in a window
    if config.get("show_image", True):
        cv2.imshow("Recording", frame)

    # Exit recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        signal_handler(signal.SIGINT, None)
