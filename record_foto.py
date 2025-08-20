import cv2
import signal
import sys
import os
import time

import utils


config = utils.get_config_for_file(__file__)

# Create output directory for frames
output_dir = "recording/foto"
os.makedirs(output_dir, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(config.get("camera_index", 0))
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)


def signal_handler(sig, frame):  # Signal handler for graceful exit
    print("\nExiting. Images saved in '{}'".format(output_dir))
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

print("Press any key to capture an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF  # Wait for key press

    if key == ord('q'):
        break  # Exit loop if 'q' is pressed
    elif key != 255:  # Any other key pressed
        frame_name = f'frame_{time.time()}'.replace(".", "_") + '.png'
        frame_filename = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
