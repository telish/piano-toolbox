import cv2
import numpy as np
import json


# Keyboard 3-D coordinates
W = 122  # Keyboard width is 122 cm
NUM_WHITE_KEYS = 52
WHITE_KEY_WIDTH = W / NUM_WHITE_KEYS
BLACK_KEY_WIDTH = 0.95  # 0.95 cm
BLACK_KEY_HEIGHT = 9  # 9 cm
WHITE_KEY_HEIGTH = 15

keyboard = np.array([
    [0,  0,  0],  # top left
    [W,  0,  0],  # top right
    [W,  WHITE_KEY_HEIGTH,  0],  # bottom right
    [0,  WHITE_KEY_HEIGTH,  0]   # bottom left
], dtype=np.float32)


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


def draw_polygon(img, points):
    img_points = np.int32(points)
    img_points = img_points.reshape((-1, 1, 2))
    cv2.polylines(img, [img_points], isClosed=True,
                  color=(0, 255, 0), thickness=2)


def calibrate_3d():
    with open("calibration/keyboard/keyboard_coords.txt", "r") as file:
        points = [tuple(map(int, line.strip("()\n").split(", ")))
                  for line in file]

    top_left, top_right, bottom_left, bottom_right = sort_points(points)

    # pixel coordinates
    image_points = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

    with open("calibration/checkerboard/camera_params.json", "r") as f:
        data = json.load(f)

    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])

    # PnP-Schätzung durchführen (RANSAC für robustere Lösung)
    success, rvec, tvec = cv2.solvePnP(
        keyboard, image_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )

    R, _ = cv2.Rodrigues(rvec)

    return mtx, dist, rvec, tvec, R


def white_key_coords_3d(white_key_index):
    """Returns the 3D coordinates of a white key given its index. The unit is cm. 
    The z-axis is always 0. (0, 0, 0) is the bottom left corner of the lowest key."""
    x_left = white_key_index * WHITE_KEY_WIDTH
    x_right = x_left + WHITE_KEY_WIDTH
    return np.array([
        [x_left, 0, 0],
        [x_left, WHITE_KEY_HEIGTH, 0],
        [x_right, WHITE_KEY_HEIGTH, 0],
        [x_right, 0, 0]
    ], dtype=np.float32)


white_keys = [
    21, 23, 24, 26, 28, 29, 31,  # lowest octave
    33, 35, 36, 38, 40, 41, 43,  # second octave
    45, 47, 48, 50, 52, 53, 55,  # third octave
    57, 59, 60, 62, 64, 65, 67,  # fourth octave
    69, 71, 72, 74, 76, 77, 79,  # fifth octave
    81, 83, 84, 86, 88, 89, 91,  # sixth octave
    93, 95, 96, 98, 100, 101, 103,  # seventh octave
    105, 107, 108  # highest octave
]
black_keys = [key for key in range(21, 108) if key not in white_keys]


def pitch_class(midi_pitch):
    c = midi_pitch % 12
    pitch_clases = ['C', 'C#', 'D', 'D#', 'E',
                    'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return pitch_clases[c]


def black_key_coords_3d(midi_pitch):
    white_below_index = white_keys.index(midi_pitch - 1)
    white_outline = white_key_coords_3d(white_below_index)
    white_x_right = white_outline[2][0]
    black_x_left = white_x_right - BLACK_KEY_WIDTH / 2
    if pitch_class(midi_pitch) in ['C#', 'F#']:
        black_x_left -= BLACK_KEY_WIDTH / 3
    if pitch_class(midi_pitch) in ['D#', 'A#']:
        black_x_left += BLACK_KEY_WIDTH / 3
    black_x_right = black_x_left + BLACK_KEY_WIDTH
    return np.array([
        [black_x_left, 0, 0],
        [black_x_left, BLACK_KEY_HEIGHT, 0],
        [black_x_right, BLACK_KEY_HEIGHT, 0],
        [black_x_right, 0, 0]
    ], dtype=np.float32)


def key_coords_3d(midi_pitch):
    if midi_pitch in white_keys:
        return white_key_coords_3d(white_keys.index(midi_pitch))
    elif midi_pitch in black_keys:
        return black_key_coords_3d(midi_pitch)
    else:
        raise ValueError(f"Invalid MIDI pitch: {midi_pitch}")


def pixel_coordinates_of_key(midi_pitch):
    """Projects the 3D coordinates of a key onto the 2D image plane to get the pixel coordinates."""
    outline = key_coords_3d(midi_pitch)
    image_points, _ = cv2.projectPoints(outline, rvec, tvec, mtx, dist)
    return image_points


def draw_key(img, midi_pitch):
    image_points = pixel_coordinates_of_key(midi_pitch)
    draw_polygon(img, image_points)
    return img


mtx, dist, rvec, tvec, R = calibrate_3d()


def main():
    image_path = "frame_1739981121_285714.png"
    img = cv2.imread(image_path)

    for midi_pitch in range(21, 109):
        outline = key_coords_3d(midi_pitch)
        if outline is not None:
            image_points, _ = cv2.projectPoints(outline, rvec, tvec, mtx, dist)
            draw_polygon(img, image_points)

    cv2.imshow("Draw Keyboard", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
