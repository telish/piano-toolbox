import cv2
import numpy as np

import utils

with open("calibration/keyboard/keyboard_coords.txt", "r") as file:
    points = [tuple(map(int, line.strip("()\n").split(", "))) for line in file]


def sort_points(points):
    sorted_by_y = sorted(points, key=lambda p: p[1])
    # Two points with the smallest Y (higher line)
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]

    # Sort top points by x-coordinate
    top_points_sorted = sorted(top_points, key=lambda p: p[0])
    top_left = top_points_sorted[0]
    top_right = top_points_sorted[1]

    # Sort bottom points by x-coordinate
    bottom_points_sorted = sorted(bottom_points, key=lambda p: p[0])
    bottom_left = bottom_points_sorted[0]
    bottom_right = bottom_points_sorted[1]

    return top_left, top_right, bottom_left, bottom_right


def compute_homography():
    top_left, top_right, bottom_left, bottom_right = sort_points(points)
    src_points = np.float32([bottom_left, top_left, top_right, bottom_right])
    dst_points = np.float32([(0, 0), (0, 1), (1, 1), (1, 0)])
    H, _ = cv2.findHomography(src_points, dst_points)
    return H


H = compute_homography()
H_inv = np.linalg.inv(H)

image_path = utils.get_keyboard_image_path()
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image!")
    exit()


def draw_trapezoid(img):
    keyboard_outline = [(0, 0), (0, 1), (1, 1), (1, 0)]
    transformed = logical_to_pixel(keyboard_outline)
    cv2.polylines(img, [transformed], isClosed=True,
                  color=(255, 0, 255), thickness=2)


def white_key_coords_logical(key):
    NUM_WHITE_KEYS = 52
    x_left = key / NUM_WHITE_KEYS
    x_right = x_left + 1.0 / NUM_WHITE_KEYS
    return [(x_left, 0), (x_left, 1), (x_right, 1), (x_right, 0)]


def black_key_right_of_logical(white_key):
    NUM_WHITE_KEYS = 52
    # https://www.reddit.com/r/piano/comments/wkofm/what_are_the_dimension_of_the_piano_keys/
    # Black key = 0.95 cm wide. White key = 2.35 cm wide
    RATIO_BLACK_WHITE = 0.95 / 2.35
    black_key_width = 1.0 / NUM_WHITE_KEYS * RATIO_BLACK_WHITE
    # Black key = 9 cm long. White key 15 cm long
    BLACK_KEY_HEIGHT_RATIO = 9.0 / 15.0
    x_left = (white_key + 1) / NUM_WHITE_KEYS - 0.5 * black_key_width
    x_right = x_left + black_key_width
    return [(x_left, 1.0 - BLACK_KEY_HEIGHT_RATIO), (x_left, 1), (x_right, 1), (x_right, 1.0 - BLACK_KEY_HEIGHT_RATIO)]


def logical_to_pixel(coord_list):
    """coord_list is a list of tuples in logical keyboard coordinates like 
    [(0, 0), (0, 1), (1, 1), (1, 0)] for the entire keyboard outline"""
    transformed = cv2.perspectiveTransform(
        np.array(coord_list).reshape(-1, 1, 2).astype(np.float32), H_inv)
    transformed = np.int32(transformed)
    return transformed


def white_key_name(key):
    offset = key % 7
    return chr(ord('A') + offset)


def draw_black_key_right_of(img, white_key):
    outline = black_key_right_of_logical(white_key)
    transformed = logical_to_pixel(outline)
    cv2.polylines(img, [transformed], isClosed=True,
                  color=(255, 0, 255), thickness=2)


def draw_white_key(img, key):
    outline = white_key_coords_logical(key)
    transformed = logical_to_pixel(outline)
    cv2.polylines(img, [transformed], isClosed=True,
                  color=(255, 0, 255), thickness=2)


def main():
    cv2.namedWindow("Draw Keyboard")

    while True:
        img_copy = image.copy()
        img_copy = utils.flip_image(img_copy)
        draw_trapezoid(img_copy)
        for white_key in range(52):
            draw_white_key(img_copy, white_key)
            name = white_key_name(white_key)
            if name in ['A', 'C', 'D', 'F', 'G'] and white_key != 51:
                draw_black_key_right_of(img_copy, white_key)
        cv2.imshow("Draw Keyboard", img_copy)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):  # Reset
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
