# Piano Toolbox for Gesture and Movement


## Calibrate

Record fotos of the keyboard and a checkerboard hat you move to different places. Put the foto of the keyboard into the directory `calibration/keyboard`. Put the fotos of the checkerboard into the directory `calibration/checkerboard`.

Perform the following steps (sequence order matters):
1. Use `calibrate_camera_orientation.py`. The keyboard should be located at the top of the image. Left hand on the left side of the image and vice versa for the right. This creates the file `calibration/camera_orientation.json`.
2. Use `calibrate_checkerboard.py`. It will use the checkerboard fotos from the directory `calibration/checkerboard` (see above) to determine the camera pareters, which it stores to `calibration/checkerboard/camara_params.json`.
3. Use `calibrate_keyboard.py` and mark the four edges of the piano keyboard in the image under `calibration/keyboard` (see above).

You will need a 10x7 checkerboard image, which you can obtain from here: https://markhedleyjones.com/projects/calibration-checkerboard-collection


## Record

- record.py: Records video, MIDI and audio simultaneously and saves the result to the directory `recording/`. It uses the scripts record_video.py, record_midi.py and record_audio.py. 
- record_audio.py: Uses the default input microphone. Change in "System Settings->Sound". 


## Simulate

Hand landmarks: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker



