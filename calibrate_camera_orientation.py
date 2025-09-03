import argparse
import json
import os
from typing import Any

import cv2
import numpy.typing as npt

import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera orientation from recording or live camera feed")
    input_group = parser.add_mutually_exclusive_group(required=False)  # Changed to False
    input_group.add_argument("--recording", type=str, help="Path to a recording")
    input_group.add_argument("--live", type=int, help="Camera index for live feed (default: 0)")
    args = parser.parse_args()

    # If neither recording nor live is specified, default to live with index 0
    if args.recording is None and args.live is None:
        args.live = 0

    return args


flip_horizontal = False
flip_vertical = False


def save_orientation() -> None:
    print("Saving camera orientation to calibration/camera_orientation.json")
    json_path = utils.retrieve_camera_orientation_file_path()
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"flip_horizontal": flip_horizontal, "flip_vertical": flip_vertical}, f)


def main() -> None:
    global flip_horizontal, flip_vertical

    args = parse_args()
    image = None
    cap = None

    if args.recording:
        utils.set_calibration_base_dir(os.path.abspath(args.recording))
        video_path = os.path.join(os.path.abspath(args.recording), "video", "recording.avi")
        c = cv2.VideoCapture(video_path)
        if not c.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            exit()
        ret, image = c.read()
        if not ret:
            print(f"Error: Could not read frame from video: {video_path}")
            exit()
    elif args.live is not None:
        cap = cv2.VideoCapture(args.live)
        if not cap.isOpened():
            print(f"Error: Could not open camera: {args.live}")
            exit()
        ret, image = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            exit()
    assert image is not None

    text = (
        "Press 'h' to toggle horizontal flip, 'v' to toggle vertical flip, 'q' to save and quit. "
        "Desired result: (1) The keyboard appears at the top of the image. "
        "(2) The left hand appears on the left side of the image, the right hand on the right."
    )

    cv2.namedWindow("Keyboard View")

    while True:
        img_copy = image.copy()
        if flip_horizontal:
            img_copy = cv2.flip(img_copy, 1)
        if flip_vertical:
            img_copy = cv2.flip(img_copy, 0)

        img_copy = utils.add_text_to_image(img_copy, text)
        cv2.imshow("Keyboard View", img_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            save_orientation()
            break
        elif key == ord("h"):
            flip_horizontal = not flip_horizontal
        elif key == ord("v"):
            flip_vertical = not flip_vertical

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
