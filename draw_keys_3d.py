"""Draw piano keys in 3D perspective on an image using homography."""

import json
import os

import cv2
import numpy as np

import keyboard_geometry
import utils
from datatypes import CorrespondingPoints, Image

homography_matrix = None  # Homography matrix


def re_init(keypoint_mappings_param: list[CorrespondingPoints] | None = None) -> None:
    """
    Initialize global calibration variables by performing 3D calibration.
    If no keypoint mappings are provided, it will load them from a file.

    Args:
        keypoint_mappings (list): A list of dictionaries containing
            "pixel" and "object" keypoints.
    """
    global homography_matrix

    keypoint_mappings = keypoint_mappings_param
    if keypoint_mappings is None:
        json_path = utils.get_keyboard_geometry_file_path()
        assert json_path
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                keypoint_mappings = json.load(file)["keypoint_mappings"]
        except FileNotFoundError:
            km: list[CorrespondingPoints] = [
                {"pixel": (100, 100), "object": (0.0, 0.0)},
                {"pixel": (1500, 100), "object": (keyboard_geometry.KEYBOARD_WIDTH, 0)},
                {"pixel": (1500, 200), "object": (keyboard_geometry.KEYBOARD_WIDTH, keyboard_geometry.WHITE_HEIGHT)},
                {"pixel": (100, 200), "object": (0.0, keyboard_geometry.WHITE_HEIGHT)},
            ]
            keypoint_mappings = km
            assert keypoint_mappings is not None

    object_points = []
    image_points = []

    assert keypoint_mappings is not None, "Keypoint mappings must be provided or loaded from file."
    assert len(keypoint_mappings) >= 4, "At least 4 point correspondences are required."
    for c in keypoint_mappings:
        object_coords = c["object"]
        pixel_coords = c["pixel"]
        object_points.append(object_coords)
        image_points.append(pixel_coords)

    src_pts = np.array(object_points, dtype=np.float32)
    dst_pts = np.array(image_points, dtype=np.float32)

    # Calculate homography with RANSAC for robustness
    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


re_init()  # Call this again if the calibration has changed and you want to update the homography matrix.


def _draw_polygon(img: Image, points: Image, color: tuple[int, int, int]) -> None:
    img_points = np.round(points).astype(np.int32)
    img_points = img_points.reshape((-1, 1, 2))
    cv2.polylines(img, [img_points], isClosed=True, color=color, thickness=1)


def pixel_coordinates_of_key(midi_pitch: int) -> Image:
    """Projects the 3D coordinates of a key onto the 2D image plane to get the pixel coordinates."""
    points = keyboard_geometry.key_points(midi_pitch)
    assert homography_matrix is not None, "Homography matrix not initialized. Call init() first."
    # For homography we need 2D points in the form (n,1,2)
    points_2d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(points_2d, homography_matrix)
    return image_points


def pixel_coordinates_of_bounding_box(midi_pitch: int) -> Image:
    """Projects the 3D coordinates of a key's bounding box onto the 2D image plane to get the pixel coordinates."""
    points = keyboard_geometry.key_bounding_box(midi_pitch)
    assert homography_matrix is not None, "Homography matrix not initialized. Call init() first."
    # For homography we need 2D points in the form (n,1,2)
    points_2d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(points_2d, homography_matrix)
    return image_points


def draw_key(
    img: Image,
    midi_pitch: int,
    color: tuple[int, int, int],
    annotation: str = "",
) -> Image:
    image_points = pixel_coordinates_of_key(midi_pitch)
    _draw_polygon(img, image_points, color)
    draw_annotation(img, midi_pitch, color, annotation, image_points)
    return img


def draw_keyboard(img: Image, color: tuple[int, int, int], outline_only: bool = False) -> Image:
    if outline_only:
        ps = keyboard_geometry.KEYBOARD_OUTLINE
        points = [
            ps["top-left"],
            ps["top-right"],
            ps["bottom-right"],
            ps["bottom-left"],
        ]
        assert homography_matrix is not None, "Homography matrix not initialized. Call init() first."
        points_2d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        image_points = cv2.perspectiveTransform(points_2d, homography_matrix)
        _draw_polygon(img, image_points, color)
        return img
    else:
        for midi_pitch in range(21, 109):
            draw_key(img, midi_pitch, color)
        return img


def draw_annotation(
    img: Image,
    midi_pitch: int,
    color: tuple[int, int, int],
    annotation: str,
    image_points: Image,
) -> None:
    if annotation:
        if midi_pitch in keyboard_geometry.BLACK_KEYS:
            y_offset = 30
        else:
            y_offset = 10

        (text_width, text_height), _ = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

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


def main() -> None:
    image_path = utils.get_keyboard_image_file_path()
    img = None
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
    if img is None:
        img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    img = utils.flip_image(img)

    for midi_pitch in range(21, 109):
        draw_key(img, midi_pitch, (0, 200, 0), f"{midi_pitch}")

    cv2.imshow("Draw Keyboard", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
