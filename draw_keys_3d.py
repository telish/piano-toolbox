import json
import os

import cv2
import numpy as np

import utils
import keyboard_geometry


H = None  # Homography matrix


def init(keypoint_mappings=None):
    """
    Initialize global calibration variables by performing 3D calibration. If no keypoint mappings are provided, it will load them from a file.

    Args:
        keypoint_mappings (list): A list of dictionaries containing "pixel" and "object" keypoints.
    """
    global H

    if keypoint_mappings is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(
            script_dir, "calibration", "keyboard", "keyboard_geometry.json"
        )
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Calibration file not found: {json_path}")
        with open(json_path, "r") as file:
            keypoint_mappings = json.load(file)["keypoint_mappings"]

    object_points = []
    image_points = []
    for c in keypoint_mappings:
        object_coords = c["object"]
        pixel_coords = c["pixel"]
        object_points.append(object_coords)
        image_points.append(pixel_coords)

    src_pts = np.array(object_points, dtype=np.float32)
    dst_pts = np.array(image_points, dtype=np.float32)

    # Calculate homography with RANSAC for robustness
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


def draw_polygon(img, points, color):
    img_points = np.round(points).astype(np.int32)
    img_points = img_points.reshape((-1, 1, 2))
    cv2.polylines(img, [img_points], isClosed=True, color=color, thickness=1)


def pixel_coordinates_of_key(midi_pitch):
    """Projects the 3D coordinates of a key onto the 2D image plane to get the pixel coordinates."""
    points = keyboard_geometry.key_points(midi_pitch)
    assert H is not None, "Homography matrix not initialized. Call init() first."
    # For homography we need 2D points in the form (n,1,2)
    points_2d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(points_2d, H)
    return image_points


def pixel_coordinates_of_bounding_box(midi_pitch):
    """Projects the 3D coordinates of a key's bounding box onto the 2D image plane to get the pixel coordinates."""
    points = keyboard_geometry.key_bounding_box(midi_pitch)
    assert H is not None, "Homography matrix not initialized. Call init() first."
    # For homography we need 2D points in the form (n,1,2)
    points_2d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(points_2d, H)
    return image_points


def draw_key(img, midi_pitch, color, annotation=""):
    image_points = pixel_coordinates_of_key(midi_pitch)
    draw_polygon(img, image_points, color)
    draw_annotation(img, midi_pitch, color, annotation, image_points)
    return img


def draw_keyboard(img, color):
    for midi_pitch in range(21, 109):
        draw_key(img, midi_pitch, color)
    return img


def draw_annotation(img, midi_pitch, color, annotation, image_points):
    if annotation:
        if midi_pitch in keyboard_geometry.black_keys:
            y_offset = 30
        else:
            y_offset = 10

        (text_width, text_height), baseline = cv2.getTextSize(
            annotation, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Find minimum and maximum x-value
        x_values = [point[0][0] for point in image_points]
        x_min = min(x_values)
        x_max = max(x_values)

        # Calculate midpoint
        x = int((x_min + x_max) / 2 - text_width / 2)
        y = int(image_points[0][0][1]) - y_offset

        # Add white background rectangle
        bg_padding = 2
        bg_x1 = x - bg_padding
        bg_y1 = y - text_height - bg_padding
        bg_x2 = x + text_width + bg_padding
        bg_y2 = y + bg_padding
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)

        # Draw text
        cv2.putText(
            img,
            annotation,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def main():
    init()
    image_path = utils.get_keyboard_image_path()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unable to load: {image_path}")
    img = utils.flip_image(img)

    for midi_pitch in range(21, 109):
        draw_key(img, midi_pitch, (0, 200, 0), f"{midi_pitch}")

    cv2.imshow("Draw Keyboard", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
