import cv2
import numpy as np
import json
import utils
import keyboard_geometry

HOMOGRAPHY = True

# Global variables for camera calibration
mtx = None
dist = None
rvec = None
tvec = None
R = None

H = None  # Homography matrix


def init(correspondences=None):
    """
    Initialize global calibration variables by performing 3D calibration. If no correspondences are provided, it will load them from a file.

    Args:
        correspondences (list): A list of dictionaries containing "pixel" and "object" keypoints.
    """
    global mtx, dist, rvec, tvec, R, H

    if correspondences is None:
        with open("calibration/keyboard/keyboard_coords.json", "r") as file:
            correspondences = json.load(file)

    # Teil 1: 3D-Kalibrierung
    object_points = []
    image_points = []
    for c in correspondences:
        object_coords = c["object"]
        pixel_coords = c["pixel"]
        object_points.append(object_coords)
        image_points.append(pixel_coords)

    # pixel coordinates
    image_points = np.array([image_points], dtype=np.float32)
    object_points_3d = np.array([[point[0], point[1], 0]
                                for point in object_points], dtype=np.float32)
    object_points_3d = object_points_3d.reshape(-1, 3)

    with open("calibration/checkerboard/camera_params.json", "r") as f:
        camera_params = json.load(f)
        mtx = np.array(camera_params["camera_matrix"])
        dist = np.array(camera_params["distortion_coefficients"])

    # Perform PnP estimation (RANSAC for more robust solution)
    success, rvec, tvec = cv2.solvePnP(
        object_points_3d, image_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )

    R, _ = cv2.Rodrigues(rvec)

    # Teil 2: Homographie-Berechnung
    # Extrahiere Modell- und Bildkoordinaten als 2D-Punkte
    src_pts = np.array(object_points, dtype=np.float32)  # 2D-Punkte
    dst_pts = np.array(image_points[0], dtype=np.float32)  # 2D-Punkte

    # Berechne Homographie mit RANSAC für Robustheit
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return mtx, dist, rvec, tvec, R, H


def draw_polygon(img, points, color):
    img_points = np.int32(points)
    img_points = img_points.reshape((-1, 1, 2))
    cv2.polylines(img, [img_points], isClosed=True,
                  color=color, thickness=1)


def make_3d(points):
    """Converts keyboard_geometry points to 3D coordinates array by adding a 0 z-coordinate."""
    coords_3d = []
    for point in points:
        coords_3d.append([point[0], point[1], 0])

    return np.array(coords_3d, dtype=np.float32)


def pixel_coordinates_of_key(midi_pitch):
    """Projects the 3D coordinates of a key onto the 2D image plane to get the pixel coordinates."""
    points = keyboard_geometry.key_points(midi_pitch)

    if HOMOGRAPHY:
        assert H is not None, "Homography matrix not initialized. Call init() first."
        # Für Homographie brauchen wir 2D-Punkte in der Form (n,1,2)
        points_2d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        image_points = cv2.perspectiveTransform(points_2d, H)
    else:
        assert rvec is not None, "Calibration not initialized. Call init() first."
        # Für 3D-Projektion brauchen wir 3D-Punkte
        outline = make_3d(points)
        image_points, _ = cv2.projectPoints(outline, rvec, tvec, mtx, dist)

    return image_points


def pixel_coordinates_of_bounding_box(midi_pitch):
    """Projects the 3D coordinates of a key's bounding box onto the 2D image plane to get the pixel coordinates."""
    points = keyboard_geometry.key_bounding_box(midi_pitch)

    if HOMOGRAPHY:
        assert H is not None, "Homography matrix not initialized. Call init() first."
        # Für Homographie brauchen wir 2D-Punkte in der Form (n,1,2)
        points_2d = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        image_points = cv2.perspectiveTransform(points_2d, H)
    else:
        assert rvec is not None, "Calibration not initialized. Call init() first."
        # Für 3D-Projektion brauchen wir 3D-Punkte
        outline = make_3d(points)
        image_points, _ = cv2.projectPoints(outline, rvec, tvec, mtx, dist)

    return image_points


def draw_key(img, midi_pitch, color, annotation=''):
    image_points = pixel_coordinates_of_key(midi_pitch)
    draw_polygon(img, image_points, color)
    draw_anotation(img, midi_pitch, color, annotation, image_points)
    return img


def draw_keyboard(img, color):
    for midi_pitch in range(21, 109):
        draw_key(img, midi_pitch, color)
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

        # Find minimum and maximum x-value
        x_values = [point[0][0] for point in image_points]
        x_min = min(x_values)
        x_max = max(x_values)

        # Calculate midpoint
        x = (x_min + x_max) / 2 - text_width / 2
        y = int(image_points[0][0][1]) - y_offset

        cv2.putText(img, annotation, (int(x), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    init()
    image_path = utils.get_keyboard_image_path()
    img = cv2.imread(image_path)
    img = utils.flip_image(img)

    for midi_pitch in range(21, 109):
        draw_key(img, midi_pitch, (0, 200, 0), f'{midi_pitch}')

    cv2.imshow("Draw Keyboard", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
