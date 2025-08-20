import json
import os

import cv2

import utils

image_path = utils.get_keyboard_image_path()
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image!")
    exit()

flip_horizontal = False
flip_vertical = False


def save_orientation():
    print("Saving camera orientation to calibration/camera_orientation.json")
    output_dir = "calibration/"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "camera_orientation.json"), "w") as f:
        json.dump({"flip_horizontal": flip_horizontal,
                    "flip_vertical": flip_vertical}, f)


def main():
    global flip_horizontal, flip_vertical
    print("Press 'h' to toggle horizontal flip, 'v' to toggle vertical flip, 'q' to save and quit.")
    print("The keyboard should be located at the top of the image. Left hand on the left side of the image and vice versa for the right.")

    cv2.namedWindow("Keyboard View")

    while True:
        img_copy = image.copy()
        if flip_horizontal:
            img_copy = cv2.flip(img_copy, 1)
        if flip_vertical:
            img_copy = cv2.flip(img_copy, 0)
        cv2.imshow("Keyboard View", img_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Reset
            save_orientation()
            break
        elif key == ord('h'):
            flip_horizontal = not flip_horizontal
        elif key == ord('v'):
            flip_vertical = not flip_vertical

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
