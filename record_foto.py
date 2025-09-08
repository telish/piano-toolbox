import argparse
import os
import signal
import sys
import time

import cv2

import utils

parser = argparse.ArgumentParser(description="Capture photos from camera.")
parser.add_argument(
    "--camera-index",
    type=int,
    default=0,
    help="Index of the camera to use (default: 0)",
)
args = parser.parse_args()

camera_index = args.camera_index

# Create output directory for ths fotos
output_dir = "recording/foto"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)


def signal_handler(_sig: int, _frame: object) -> None:  # Signal handler for graceful exit
    print("\nExiting. Images saved in '{}'".format(output_dir))
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

print("Press any key to capture an image. Press 'q' to quit.")

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_output = img.copy()

    text = "Press any key to capture an image. Press 'q' to quit."
    utils.add_text_to_image(frame_output, text)

    cv2.imshow("Camera", frame_output)
    key = cv2.waitKey(1) & 0xFF  # Wait for key press

    if key == ord("q"):
        break  # Exit loop if 'q' is pressed
    elif key != 255:  # Any other key pressed
        frame_name = f"frame_{time.time()}".replace(".", "_") + ".png"
        frame_filename = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_filename, img)
        print(f"Saved: {frame_filename}")

cap.release()
cv2.destroyAllWindows()
