import os

def get_keyboard_image_path():
    # Load first PNG image from calibration/keyboard directory
    keyboard_dir = "calibration/keyboard"
    png_files = [f for f in os.listdir(keyboard_dir) if f.lower().endswith(".png")]
    if not png_files:
        print("Error: No PNG files found in calibration/keyboard!")
        exit()
    image_path = os.path.join(keyboard_dir, sorted(png_files)[0])
    return image_path
