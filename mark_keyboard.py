import os

import cv2
import numpy as np

# Load image
image_path = "recording/foto/frame_1739987266_919347.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image!")
    exit()

points = []  # Store trapezoid points


def draw_trapezoid(img, points):
    """Draw the trapezoid based on selected points."""
    if len(points) > 1:
        cv2.polylines(img, [np.array(points, np.int32)],
                      isClosed=True, color=(255, 0, 255), thickness=2)

    for point in points:
        cv2.circle(img, point, 10, (255, 0, 255), -1)


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        # If less than 4 points, add a new one
        if len(points) < 4:
            points.append((x, y))


def save_coords(points):
    print("saving", len(points))
    if len(points) == 4:
        output_dir = "calibration/3d/"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "keyboard_coords.txt"), "w") as f:
            for p in points:
                f.write(f"({p[0]}, {p[1]})\n")
        print("Coordinates saved to keyboard_coords.txt")


def main():
    global points

    cv2.namedWindow("Draw Keyboard")
    cv2.setMouseCallback("Draw Keyboard", mouse_callback)

    while True:
        img_copy = image.copy()
        draw_trapezoid(img_copy, points)
        cv2.imshow("Draw Keyboard", img_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Reset
            save_coords(points)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
