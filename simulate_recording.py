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
interactive_mode = config["interactive_mode"]

video_path = os.path.join(recording_base, "video")


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


def draw_text(img, instruction_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(instruction_text, font, 0.7, 2)[0]

    # Position at bottom of the image
    text_x = 10
    text_y = img.shape[0] - 20

    # Draw background rectangle
    cv2.rectangle(img,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 0, 0), -1)

    # Draw text
    cv2.putText(img, instruction_text,
                (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Update the display with the text
    cv2.imshow('Simulate Recording', img)


def handle_keyboard_input(img):
    global skip_to_next_note
    stop = interactive_mode and \
        ((not skip_to_next_note['active']) or
         (skip_to_next_note['active'] and skip_to_next_note['note_received']))
    if stop:
        # Draw instructions directly on the image
        instruction_text = "Press 'f' for next frame, 'n' for the next note, 'q' to quit"
        draw_text(img, instruction_text)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('f'):
                break
            if key == ord('n'):
                skip_to_next_note = {'active': True,
                                     'note_received': False, 'next_frame': False}
                break
            elif key == ord('q'):
                print("Quitting simulation.")
                cv2.destroyAllWindows()
                exit(0)
    else:
        cv2.waitKey(1)  # Just to update the window without blocking


def get_all_events(recording_base, parse_midi_mgs, parse_video):
    midi_events = parse_midi_mgs(os.path.join(
        recording_base, "midi/midi_msg.txt"))
    video_events = parse_video(video_path)

    # Combine and sort events by timestamp
    all_events = midi_events + video_events
    all_events.sort(key=lambda event: event['timestamp'])

    return all_events


skip_to_next_note = {'active': False, 'note_received': False}


def process_midi(event):
    global skip_to_next_note
    print(f"{event['timestamp']}: {event['message']}")
    analysis_hub.process_midi_event(event)
    res = analysis_hub.last_midi_result
    if analysis_hub.last_midi_result["msg.type"] == "note_on":
        print(res['hand'].capitalize(), res['finger'])
        if skip_to_next_note['active']:
            skip_to_next_note['note_received'] = True


class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = None

    def load_video_if_needed(self):
        if self.video_capture is None:
            video_file = os.path.join(self.video_path, 'recording.avi')
            self.video_capture = cv2.VideoCapture(video_file)

    def read_frame(self, frame_number):
        self.load_video_if_needed()
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = self.video_capture.read()
        return ret, img


def process_video_frame(event, video_processor):
    print(f"{event['timestamp']:.7f}: Frame {event['frame_number']}")

    # Read frame
    ret, img = video_processor.read_frame(event['frame_number'])
    if not ret:
        print(f"Failed to read frame {event['frame_number']}")
        return

    # Process frame
    img = utils.flip_image(img)
    analysis_hub.process_frame(img)
    img = analysis_hub.last_mp_image

    # Draw notes
    for midi_pitch in analysis_hub.current_notes.keys():
        if analysis_hub.current_notes[midi_pitch]["hand"] == "left":
            color = (0, 0, 200)  # Red color
        elif analysis_hub.current_notes[midi_pitch]["hand"] == "right":
            color = (0, 200, 0)  # Green color
        else:
            color = (200, 200, 0)  # Yellow for unknown hand

        annotation = f"{analysis_hub.current_notes[midi_pitch]['finger']}"
        img = draw_keys_3d.draw_key(img, midi_pitch, color, annotation)

    cv2.imshow('Simulate Recording', img)
    handle_keyboard_input(img)


video_player = VideoPlayer(video_path)
all_events = get_all_events(recording_base, parse_midi_mgs, parse_video)
start_real = time.time()
start_recording = all_events[0]['timestamp']
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
        process_midi(event)
    elif event['type'] == 'video':
        process_video_frame(event, video_player)
    else:
        assert False, f"Unknown event type: {event['type']}"

# Clean up
cv2.destroyAllWindows()
