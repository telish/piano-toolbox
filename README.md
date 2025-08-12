# Checkerboard

https://markhedleyjones.com/projects/calibration-checkerboard-collection


# record.py

Records video, MIDI and audio simultaneously and saves the result to the directory `recording/`. It uses the scripts record_video.py, record_midi.py and record_audio.py.


# record_audio.py

Uses the default input microphone. Change in "System Settings->Sound". 


# Hand Landmarks
see https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker


# calibrate_checkerboard.py

Calibrates the camera using a 10x7 checkerboard image. Make several photos using record_foto.py and put them into a directory `calibration/checkerboard` before calling the script. The results will be written to `calibration/camera_params.json`. 


# mark_keyboard.py

Mark the edges of the keyboard to allow translation of MIDI notes to key coordinates in the image. The result will be written to `calibration/3d/keyboard_coords.txt`.



