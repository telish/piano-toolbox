import json
import os

import cv2

from datatypes import Image

_calibration_base_dir = os.path.dirname(os.path.abspath(__file__))


def set_calibration_base_dir(dir: str) -> None:
    global _calibration_base_dir
    _calibration_base_dir = dir
    load_flip_settings()


def get_keyboard_image_file_path() -> str:
    """Get the path to the keyboard image.

    Returns:
        str: The path to the keyboard image.
    """
    return os.path.join(_calibration_base_dir, "calibration", "keyboard", "foto.png")


def get_keyboard_geometry_file_path() -> str:
    return os.path.join(_calibration_base_dir, "calibration", "keyboard", "keyboard_geometry.json")


def retrieve_camera_orientation_file_path() -> str:
    return os.path.join(_calibration_base_dir, "calibration", "camera_orientation.json")


flip_horizontal, flip_vertical = False, False


def load_flip_settings() -> None:
    global flip_vertical, flip_horizontal
    json_path = os.path.join(_calibration_base_dir, "calibration", "camera_orientation.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            orientation = json.load(f)
        flip_horizontal = orientation.get("flip_horizontal", False)
        flip_vertical = orientation.get("flip_vertical", False)


load_flip_settings()


def flip_image(img: Image) -> Image:
    if flip_vertical:
        img = cv2.flip(img, 0)
    if flip_horizontal:
        img = cv2.flip(img, 1)
    return img


def add_text_to_image(
    img: Image,
    text: str,
    position: str = "bottom-left",
    padding: int = 10,
    font_scale: float = 0.7,
    thickness: int = 2,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    max_text_width: int | None = None,
) -> None:
    """
    Add multiline text to an image with background, supporting both user-defined
    and automatic line breaks based on available width.

    Args:
        img: The image to add text to
        text: A string with optional new lines ('\n') as line separators
        position: Where to place the text ('bottom-left', 'bottom-right', 'top-left', 'top-right')
        padding: Padding around text in pixels
        font_scale: Font scale factor
        thickness: Line thickness
        text_color: Text color (B,G,R)
        bg_color: Background color (B,G,R)
        max_text_width: Maximum width for a text line in pixels, None for auto (80% of image width)

    Returns:
        Image with text added
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Default max width if not specified
    if max_text_width is None:
        max_text_width = int(img.shape[1] * 0.8)  # 80% of image width

    # First split by user-defined line breaks
    user_text_lines = text.split("\n")

    # Then apply automatic word wrapping within each user-defined line
    final_text_lines = []
    for user_line in user_text_lines:
        words = user_line.split()
        if not words:
            final_text_lines.append("")
            continue

        current_line = words[0]
        for word in words[1:]:
            # Check if adding this word exceeds the max width
            test_line = current_line + " " + word
            test_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

            if test_size[0] <= max_text_width:
                current_line = test_line
            else:
                final_text_lines.append(current_line)
                current_line = word

        final_text_lines.append(current_line)

    # Calculate text block dimensions
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in final_text_lines]
    text_widths = [size[0][0] for size in text_sizes]
    text_heights = [size[0][1] for size in text_sizes]

    max_width = max(text_widths) if text_widths else 0
    total_height = sum(text_heights) + (len(final_text_lines) - 1) * padding if text_heights else 0

    # Determine position coordinates
    if position == "bottom-left":
        x = padding
        y = img.shape[0] - padding - total_height
    elif position == "bottom-right":
        x = img.shape[1] - max_width - padding
        y = img.shape[0] - padding - total_height
    elif position == "top-left":
        x = padding
        y = padding + text_heights[0] if text_heights else padding
    elif position == "top-right":
        x = img.shape[1] - max_width - padding
        y = padding + text_heights[0] if text_heights else padding
    else:
        # Default to bottom-left
        x = padding
        y = img.shape[0] - padding - total_height

    # Draw background rectangle
    bg_top_left = (x - padding, y - (text_heights[0] if text_heights else 0) - padding)
    bg_bottom_right = (x + max_width + padding, y + total_height + padding)
    cv2.rectangle(img, bg_top_left, bg_bottom_right, bg_color, -1)

    # Draw each line of text
    current_y = y
    for i, line in enumerate(final_text_lines):
        cv2.putText(
            img,
            line,
            (x, current_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        current_y += text_heights[i] + padding
