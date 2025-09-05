import argparse  # Import argparse for command-line arguments
import os
import signal
import sys
import time

import cv2

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Capture photos from camera.")
parser.add_argument(
    "--camera-index",
    type=int,
    default=0,
    help="Index of the camera to use (default: 0)",
)
args = parser.parse_args()

# Use the camera index from the command-line argument
camera_index = args.camera_index

# Create output directory for frames
output_dir = "recording/foto"
os.makedirs(output_dir, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)


def signal_handler(_sig: int, _frame: object) -> None:  # Signal handler for graceful exit
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

    frame_output = frame.copy()

    # Add text to the image
    text = "Press any key to capture an image. Press 'q' to quit."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text

    # Get text size to position it and create background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position at the bottom of the frame
    text_x = 10
    text_y = frame_output.shape[0] - 20

    # Draw dark background rectangle for better readability
    cv2.rectangle(
        frame_output,
        (text_x - 5, text_y - text_height - 5),
        (text_x + text_width + 5, text_y + 5),
        (0, 0, 0),
        -1,
    )

    # Draw text

    cv2.putText(
        frame_output,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.imshow("Camera", frame_output)
    key = cv2.waitKey(1) & 0xFF  # Wait for key press

    if key == ord("q"):
        break  # Exit loop if 'q' is pressed
    elif key != 255:  # Any other key pressed
        frame_name = f"frame_{time.time()}".replace(".", "_") + ".png"
        frame_filename = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
