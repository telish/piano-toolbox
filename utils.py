import os
import cv2
import json


def get_keyboard_image_path():
    # Load first PNG image from calibration/keyboard directory
    keyboard_dir = "calibration/keyboard"
    png_files = [f for f in os.listdir(
        keyboard_dir) if f.lower().endswith(".png")]
    if not png_files:
        print("Error: No PNG files found in calibration/keyboard!")
        exit()
    image_path = os.path.join(keyboard_dir, sorted(png_files)[0])
    return image_path


# Load flip settings from calibration/camera_orientation.json
orientation_path = "calibration/camera_orientation.json"
if not os.path.exists(orientation_path):
    print("Error: calibration/camera_orientation.json not found!")
    exit()
with open(orientation_path, "r") as f:
    orientation = json.load(f)
flip_horizontal = orientation.get("flip_horizontal", False)
flip_vertical = orientation.get("flip_vertical", False)


def flip_image(img):
    if flip_vertical:
        img = cv2.flip(img, 0)
    if flip_horizontal:
        img = cv2.flip(img, 1)
    return img


def get_config_for_file(file_name):
    file_name = os.path.basename(file_name)
    with open('config.json', 'r') as file:
        config = json.load(file)
    return config[os.path.basename(file_name)]
