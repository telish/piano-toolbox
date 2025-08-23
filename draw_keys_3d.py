import cv2
import numpy as np
import json
import utils
import keyboard_geometry


keyboard = np.array([
    [0,  0,  0],  # top left
    [keyboard_geometry.KEYBOARD_WIDTH,  0,  0],  # top right
    [keyboard_geometry.KEYBOARD_WIDTH,
        keyboard_geometry.WHITE_HEIGHT,  0],  # bottom right
    [0,  keyboard_geometry.WHITE_HEIGHT,  0]   # bottom left
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


def draw_polygon(img, points, color):
    img_points = np.int32(points)
    img_points = img_points.reshape((-1, 1, 2))
    cv2.polylines(img, [img_points], isClosed=True,
                  color=color, thickness=1)


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


def key_coords_3d(midi_pitch):
    """Converts keyboard_geometry points to 3D coordinates array."""
    points = keyboard_geometry.key_points(midi_pitch)

    # Convert to the required format: each point as [x, y, 0]
    # Adding z=0 as the third coordinate
    coords_3d = []
    for point in points:
        coords_3d.append([point[0], point[1], 0])

    return np.array(coords_3d, dtype=np.float32)


def pixel_coordinates_of_key(midi_pitch):
    """Projects the 3D coordinates of a key onto the 2D image plane to get the pixel coordinates."""
    outline = key_coords_3d(midi_pitch)
    image_points, _ = cv2.projectPoints(outline, rvec, tvec, mtx, dist)
    return image_points


def draw_key(img, midi_pitch, color, annotation=''):
    image_points = pixel_coordinates_of_key(midi_pitch)
    draw_polygon(img, image_points, color)
    draw_anotation(img, midi_pitch, color, annotation, image_points)
    return img


def draw_anotation(img, midi_pitch, color, annotation, image_points):
    if annotation:
        if midi_pitch in keyboard_geometry.black_keys:
            y_offset = 30
        else:
            y_offset = 10

        (text_width, text_height), baseline = cv2.getTextSize(
            annotation, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Finde minimalen und maximalen x-Wert
        x_values = [point[0][0] for point in image_points]
        x_min = min(x_values)
        x_max = max(x_values)

        # Berechne Mittelpunkt
        x = (x_min + x_max) / 2 - text_width / 2
        y = int(image_points[0][0][1]) - y_offset

        cv2.putText(img, annotation, (int(x), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


mtx, dist, rvec, tvec, R = calibrate_3d()


def main():
    image_path = utils.get_keyboard_image_path()
    img = cv2.imread(image_path)
    img = utils.flip_image(img)

    for midi_pitch in range(21, 109):
        # outline = key_coords_3d(midi_pitch)
        # if outline is not None:
        #     image_points, _ = cv2.projectPoints(outline, rvec, tvec, mtx, dist)
        #     draw_polygon(img, image_points, color=(0, 255, 0))
        draw_key(img, midi_pitch, (0, 255, 0), f'{midi_pitch}')

    cv2.imshow("Draw Keyboard", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
