import cv2
import signal
import sys
import os
import time

# Create output directory for frames
output_dir = "recording/video"
os.makedirs(output_dir, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("record-video.py: Error. Could not open webcam.")
    sys.exit(1)


# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("record-video.py: Frames saved in {}".format(output_dir))
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

print("record-video.py: Recording camera. Press Ctrl+C to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("record-video.py: Error. Failed to capture frame.")
        break

    frame_name = f'frame_{time.time()}'.replace(".", "_") + '.png'
    frame_filename = os.path.join(output_dir, frame_name)
    cv2.imwrite(frame_filename, frame)
    cv2.imshow('record-video.py', frame)
    cv2.pollKey()
