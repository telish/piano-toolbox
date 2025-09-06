"""
This script provides an interactive tool for calibrating the geometry of a piano keyboard from
either a video recording or a live camera feed. Users can mark the four corners of the keyboard,
adjust key lengths, and save the calibration data and annotated image for further processing.
"""

import argparse
import json
import os
from typing import List, TypedDict

import cv2
import numpy as np

import draw_keys_3d
import keyboard_geometry
import utils
from datatypes import CorrespondingPoints, Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate keyboard geometry from recording or live camera feed")
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--recording", type=str, help="Path to a recording")
    input_group.add_argument("--live", type=int, help="Camera index for live feed (default: 0)")
    args = parser.parse_args()

    # If neither recording nor live is specified, default to live with index 0
    if args.recording is None and args.live is None:
        args.live = 0

    return args


_state = {
    "user_defined_points": [],
    "dragging_index": -1,
    "drag_threshold": 10,
}


def draw_points(img: Image, points: list[CorrespondingPoints]) -> None:
    for i, point in enumerate(points):
        p = point["pixel"]
        size = 5 if i == _state["dragging_index"] else 3  # Larger point if being dragged
        color = (0, 0, 255) if i == _state["dragging_index"] else (255, 0, 255)  # Red if dragged, magenta otherwise
        cv2.rectangle(
            img,
            (int(p[0]) - size, int(p[1]) - size),
            (int(p[0]) + size, int(p[1]) + size),
            color,
            -1,
        )
        # Display point number
        cv2.putText(
            img,
            f"{i + 1}: {point['object']}",
            (int(p[0]) + 10, int(p[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_trapezoid(img: Image, points: list[CorrespondingPoints]) -> None:
    """Draw the trapezoid based on selected points."""

    if len(points) > 1:
        # For drawing the trapezoid, make sure points are in the correct order
        if len(points) == 4:
            sorted_points = get_correspondences_without_projection(points)
            ordered_points = [
                sorted_points[0]["pixel"],
                sorted_points[1]["pixel"],
                sorted_points[3]["pixel"],
                sorted_points[2]["pixel"],
            ]  # Correct drawing order
            cv2.polylines(
                img,
                [np.array(ordered_points, np.int32)],
                isClosed=True,
                color=(255, 0, 255),
                thickness=1,
            )
        else:
            # For incomplete sets, just connect in the order clicked
            all_pixel_coords = [p["pixel"] for p in _state["user_defined_points"]]
            cv2.polylines(
                img,
                [np.array(all_pixel_coords, np.int32)],
                isClosed=True,
                color=(255, 0, 255),
                thickness=1,
            )


def find_closest_point_index(x: int, y: int, points: list[tuple[int, int]], max_distance: int = 20) -> int:
    """Find the index of the closest point to position (x,y)."""
    if not points:
        return -1

    closest_idx = -1
    min_dist = float("inf")

    for i, point in enumerate(points):
        dist = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            closest_idx = i

    return closest_idx


def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if an existing point was clicked
        all_pixel_coords = [p["pixel"] for p in _state["user_defined_points"]]
        _state["dragging_index"] = find_closest_point_index(x, y, all_pixel_coords, _state["drag_threshold"])

        # If no nearby point and still room for more points
        if _state["dragging_index"] == -1:
            _state["user_defined_points"].append({"pixel": (x, y), "object": None})
            # Start dragging the new point immediately
            _state["dragging_index"] = len(_state["user_defined_points"]) - 1

    elif event == cv2.EVENT_MOUSEMOVE:
        # Drag point if one is selected
        if _state["dragging_index"] != -1 and _state["dragging_index"] < len(_state["user_defined_points"]):
            _state["user_defined_points"][_state["dragging_index"]]["pixel"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # End dragging
        _state["dragging_index"] = -1  # End dragging


def save_coords(image: Image) -> None:
    if len(_state["user_defined_points"]) >= 4:
        file_path = utils.get_keyboard_geometry_file_path()
        assert file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        add_object_coords(_state["user_defined_points"])
        result = {
            "keypoint_mappings": _state["user_defined_points"],
            "black_height": keyboard_geometry.black_height,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"Coordinates saved in {file_path}")

        cv2.imwrite(utils.get_keyboard_image_file_path(), image)
        print(f"Image saved in {utils.get_keyboard_image_file_path()}")


def add_object_coords(points: list[CorrespondingPoints]) -> None:
    for p in points:
        if p["object"] is None:
            object_coords = find_closest_point(p)
            p["object"] = object_coords


def main(recording: str | None = None, live: int | None = None) -> None:
    cap = None
    image: Image | None = None

    # If direct parameters are provided, use them
    if recording is not None or live is not None:
        args = argparse.Namespace()
        args.recording = recording
        args.live = live
    else:
        # Otherwise parse from command line
        args = parse_args()

    if args.recording:
        video_path = os.path.join(os.path.abspath(args.recording), "video", "recording.avi")
        utils.set_calibration_base_dir(os.path.abspath(args.recording))
        # Open video and read first frame
        c = cv2.VideoCapture(video_path)
        if not c.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        ret, image = c.read()
        if not ret:
            raise ValueError(f"Could not read frame from video: {video_path}")
    elif args.live is not None:
        cap = cv2.VideoCapture(args.live)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {args.live}")

    cv2.namedWindow("Draw Keyboard")

    cv2.setMouseCallback("Draw Keyboard", mouse_callback)

    instructions = (
        "Mark the 4 corners of the keyboard:\n"
        "1. Click to place a point\n"
        "2. Drag points to adjust\n"
        "3. Press '+' or '-' to adjust the black key length\n"
        "4. Press 's' to save and quit\n"
        "5. Press 'q' to quit without saving"
    )

    while True:
        if cap is not None:
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                exit(1)
            img_draw = image
        else:
            assert image is not None
            img_draw = image.copy()

        img_draw = utils.flip_image(img_draw)

        # Draw points first
        draw_points(img_draw, _state["user_defined_points"])

        # Different rendering based on number of points
        if len(_state["user_defined_points"]) < 4:
            draw_trapezoid(img_draw, _state["user_defined_points"])
        elif len(_state["user_defined_points"]) == 4:
            draw_trapezoid(img_draw, _state["user_defined_points"])
            get_correspondences_without_projection(_state["user_defined_points"])
            draw_keys_3d.re_init(_state["user_defined_points"])
            draw_keys_3d.draw_keyboard(img_draw, (0, 200, 0))
        elif len(_state["user_defined_points"]) > 4:
            add_object_coords(_state["user_defined_points"])
            draw_keys_3d.re_init(_state["user_defined_points"])
            draw_keys_3d.draw_keyboard(img_draw, (0, 200, 0))

        # Add instructions
        utils.add_text_to_image(img_draw, instructions, position="bottom-left")

        cv2.imshow("Draw Keyboard", img_draw)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Quit without saving
            break
        elif key == ord("s"):  # Save and quit
            save_coords(image)
            break
        elif key == ord("+") or key == ord("="):  # Increase black key length
            keyboard_geometry.black_height = keyboard_geometry.black_height + 0.5
            keyboard_geometry.re_init()
        elif key == ord("-") or key == ord("_"):  # Decrease black key length
            keyboard_geometry.black_height = keyboard_geometry.black_height - 0.5
            keyboard_geometry.re_init()

    cv2.destroyAllWindows()


def find_closest_point(pt: CorrespondingPoints) -> tuple[float, float]:
    min_distance = float("inf")
    closest_pitch, closest_index = None, None
    for pitch in range(21, 109):
        key_pts = draw_keys_3d.pixel_coordinates_of_key(pitch)
        # Find the closest point in this key's points
        for idx, key_pt in enumerate(key_pts):
            distance = np.linalg.norm(np.array(key_pt) - np.array(pt["pixel"]))
            if distance < min_distance:
                min_distance = distance
                # closest = key_pt
                closest_pitch = pitch
                closest_index = idx  # Store the index
    assert closest_pitch is not None and closest_index is not None, f"Could not find closest pitch for point {pt}"

    object_coords = keyboard_geometry.key_points(closest_pitch)[closest_index]

    return object_coords


def get_correspondences_without_projection(
    points: list[CorrespondingPoints],
) -> tuple[CorrespondingPoints, CorrespondingPoints, CorrespondingPoints, CorrespondingPoints]:
    assert len(points) == 4, "Exactly 4 points are required."

    sorted_by_y = sorted(points, key=lambda p: p["pixel"][1])
    # Two points with the smallest Y (higher line)
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]

    # Sort top points by x-coordinate
    top_points_sorted = sorted(top_points, key=lambda p: p["pixel"][0])
    top_left = top_points_sorted[0]
    top_right = top_points_sorted[1]

    # Sort bottom points by x-coordinate
    bottom_points_sorted = sorted(bottom_points, key=lambda p: p["pixel"][0])
    bottom_left = bottom_points_sorted[0]
    bottom_right = bottom_points_sorted[1]

    top_left["object"] = keyboard_geometry.KEYBOARD_OUTLINE["top-left"]
    top_right["object"] = keyboard_geometry.KEYBOARD_OUTLINE["top-right"]
    bottom_left["object"] = keyboard_geometry.KEYBOARD_OUTLINE["bottom-left"]
    bottom_right["object"] = keyboard_geometry.KEYBOARD_OUTLINE["bottom-right"]

    return top_left, top_right, bottom_left, bottom_right


if __name__ == "__main__":
    main()
