import json
import os
import time

import cv2
import mido

import analysis_hub
import draw_keys_3d
import utils
import osc_sender

FAST_MODE = True  # No sleeping to speed up the simulation

config = utils.get_config_for_file(__file__)
osc_sender.configure(config["port_outgoing"])
recording_base = config["path"]


def parse_midi_mgs(filename):
    """Parse MIDI messages from file. Returns empty list if file doesn't exist."""
    result = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                try:
                    timestamp, msg_text = line.strip().split(': ', 1)
                    timestamp = float(timestamp)  # Convert to float
                    # Convert text to mido Message
                    msg = mido.Message.from_str(msg_text)
                    result.append(
                        {'timestamp': timestamp, 'type': 'midi', 'message': msg})
                except ValueError as e:
                    print(f"Error parsing line: {line} -> {e}")
    except FileNotFoundError:
        print(f"MIDI file not found: {filename}")
        return []

    return result


def parse_video_timestamps(timestamps_file):
    """Parse the timestamps.json file created by record_video.py"""
    with open(timestamps_file, 'r') as f:
        timestamps = json.load(f)
    return [{'timestamp': t['timestamp'],
             'type': 'video',
             'frame_number': t['frame_number']}
            for t in timestamps]


def parse_video(path):
    """Read video file and timestamps"""
    # Read timestamps
    timestamps_file = os.path.join(path, 'timestamps.json')
    if not os.path.exists(timestamps_file):
        raise FileNotFoundError(
            f"Timestamps file not found: {timestamps_file}")

    # Open video file
    video_file = os.path.join(path, 'recording.avi')
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")

    return parse_video_timestamps(timestamps_file)


midi_events = parse_midi_mgs(os.path.join(recording_base, "midi/midi_msg.txt"))
video_path = os.path.join(recording_base, "video")
video_events = parse_video(video_path)

# Combine and sort events by timestamp
all_events = midi_events + video_events
all_events.sort(key=lambda event: event['timestamp'])
start_real = time.time()
start_recording = all_events[0]['timestamp']

# Output events in chronological order
img = None
for event in all_events:
    # Wait until the event's timestamp is reached
    time_to_sleep = event['timestamp'] - \
        start_recording - (time.time() - start_real)
    if time_to_sleep < 0:  # Make sure time to sleep is not negative.
        time_to_sleep = 0
    if not FAST_MODE:
        time.sleep(time_to_sleep)

    if event['type'] == 'midi':
        print(f"{event['timestamp']}: {event['message']}")
        analysis_hub.process_midi_event(event)
        res = analysis_hub.last_midi_result
        if analysis_hub.last_midi_result["msg.type"] == "note_on":
            print(res['hand'].capitalize(), res['finger'])

    elif event['type'] == 'video':
        print(f"{event['timestamp']:.7f}: Frame {event['frame_number']}")

        # Lazy loading of video file
        if 'video_capture' not in locals():
            video_file = os.path.join(video_path, 'recording.avi')
            video_capture = cv2.VideoCapture(video_file)

        # Seek to frame number and read frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, event['frame_number'])
        ret, img = video_capture.read()
        if not ret:
            print(f"Failed to read frame {event['frame_number']}")
            continue

        img = utils.flip_image(img)
        analysis_hub.process_frame(img)
        img = analysis_hub.last_mp_image
        for midi_pitch in analysis_hub.current_notes.keys():
            if analysis_hub.current_notes[midi_pitch]["hand"] == "left":
                color = (0, 0, 200)  # Red color
            elif analysis_hub.current_notes[midi_pitch]["hand"] == "right":
                color = (0, 200, 0)  # Green color
            else:
                color = (200, 200, 0)  # Yellow for unknown hand

            annotation = f"{analysis_hub.current_notes[midi_pitch]['finger']}"
            img = draw_keys_3d.draw_key(
                img, midi_pitch, color, annotation)

        cv2.imshow('Simulate Recording', img)
        cv2.waitKey(1)

# Clean up at the end
if 'video_capture' in locals():
    video_capture.release()
cv2.destroyAllWindows()
