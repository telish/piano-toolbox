# Piano Toolbox for Gesture and Movement Analysis


## Calibration

Capture a photo of the keyboard. Place the keyboard photo in the `calibration/keyboard` directory.

Follow these steps in sequence:
1. Run `calibrate_camera_orientation.py`. The keyboard should appear at the top of the image, with the left hand on the left side of the image and the right hand on the right. This creates the `calibration/camera_orientation.json` file.
2. Run `calibrate_keyboard.py` and mark the four corners of the piano keyboard in the image located in the `calibration/keyboard` directory.
   - Click and drag points to position them precisely to the corners of the keyboard
   - Press '+' or '-' to adjust the black key length
   - Press 'q' to save and quit


## Recording

- `record.py`: Records video, MIDI, and audio simultaneously, saving the results to the `recording/` directory. It utilizes the scripts `record_video.py`, `record_midi.py`, and `record_audio.py`.
- `record_audio.py`: Uses the default input microphone. You can change this in "System Settings -> Sound" (macOS).


## Simulation

Hand landmarks reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker



