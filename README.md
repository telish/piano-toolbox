# Piano Toolbox for Gesture and Movement Analysis


## Calibration

Capture photos of the keyboard and a checkerboard in various positions. Place the keyboard photo in the `calibration/keyboard` directory and the checkerboard photos in the `calibration/checkerboard` directory.

Follow these steps in sequence:
1. Run `calibrate_camera_orientation.py`. The keyboard should appear at the top of the image, with the left hand on the left side of the image and the right hand on the right. This creates the `calibration/camera_orientation.json` file.
2. Run `calibrate_checkerboard.py`. This uses the checkerboard photos from the `calibration/checkerboard` directory to determine the camera parameters, storing them in `calibration/checkerboard/camera_params.json`. It is advisable to disable your camera's autofocus feature and use a fixed focus on the keyboard.
3. Run `calibrate_keyboard.py` and mark the four corners of the piano keyboard in the image located in the `calibration/keyboard` directory.

You will need a 10x7 checkerboard image, which you can download from: https://markhedleyjones.com/projects/calibration-checkerboard-collection


## Recording

- `record.py`: Records video, MIDI, and audio simultaneously, saving the results to the `recording/` directory. It utilizes the scripts `record_video.py`, `record_midi.py`, and `record_audio.py`.
- `record_audio.py`: Uses the default input microphone. You can change this in "System Settings -> Sound" (macOS).


## Simulation

Hand landmarks reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker



