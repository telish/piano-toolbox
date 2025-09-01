# Piano Toolbox for Gesture and Movement Analysis


## Calibration

Capture a photo of the keyboard. Place the keyboard photo in the `calibration/keyboard` directory.

Follow these steps in sequence:
First, run `calibrate_camera_orientation.py`. The keyboard should appear at the top of the image, with the left hand on the left side of the image and the right hand on the right. This creates the `calibration/camera_orientation.json` file.

Then, run `calibrate_keyboard.py` and mark the four corners of the piano keyboard in the image located in the `calibration/keyboard` directory.

- Click and drag points to position them precisely to the corners of the keyboard
- Press '+' or '-' to adjust the black key length

**calibrate_camera_orientation.py:** 

- `--recording <path>`: Use a video file from a recording directory.
- `--image <path>`: Use a static image file.
- `--live <camera_index>`: Use a live camera feed (specify camera index).

**calibrate_keyboard.py:** 

- `--recording <path>`: Use a video file from a recording directory.
- `--image <path>`: Use a static image file.
- `--live <camera_index>`: Use a live camera feed (specify camera index).

## Recording

**record.py:** Records video, MIDI, and audio simultaneously, saving the results to the `recording/` directory. It utilizes the scripts `record_video.py`, `record_midi.py`, and `record_audio.py`. It forward arguments to the subprocesses:

- `--args-video "<video_args>"`: Arguments for `record_video.py`
- `--args-audio "<audio_args>"`: Arguments for `record_audio.py`
- `--args-midi "<midi_args>"`: Arguments for `record_midi.py`

For example, to record with a specific camera and MIDI port:

```sh
python record.py --args-video "--camera-index 1" --args-midi "--port-index 2"
```

**record_audio.py:**

- Uses the default input microphone. You can change this in "System Settings -> Sound" (macOS).

  
**record_video.py:**  

- `--camera-index <int>`: Select camera device (default: 0)
- `--show-image`: Show video frames during recording

**record_foto.py**  

- `--camera-index <int>`: Select camera device (default: 0)

**record_midi.py**  

- `--port-index <int>`: Select MIDI input port by index (default: 0)


## Simulation

Hand landmarks reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

**simulate_recording.py:**  

- `--port-out <int>`: OSC outgoing port (default: 9876)
- `--recording <path>`: Path to the recording directory (default: ./recording)



