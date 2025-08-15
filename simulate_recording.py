import json
import os
import time

import cv2
import mido
from pythonosc.udp_client import SimpleUDPClient
import numpy as np

import track_hands
import draw_keys_3d

FAST_MODE = True  # No sleeping to speed up the simulation

with open('config.json', 'r') as file:
    config = json.load(file)
config = config[os.path.basename(__file__)]

osc_client = SimpleUDPClient("127.0.0.1", config["port_outgoing"])


def parse_midi_mgs(filename):
    result = []
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
    return result


def parse_video(path):
    video_frames = []
    for filename in os.listdir(path):
        if filename.startswith("frame_") and filename.endswith(".png"):
            num_as_string = filename[6:-4].replace('_', '.')
            try:
                timestamp = float(num_as_string)
                video_frames.append(
                    {'timestamp': timestamp, 'type': 'video', 'filename': filename})
            except ValueError as e:
                print(f"Error parsing filename: {filename} -> {e}")
    return video_frames


def get_hand_positions(result):
    left_pos = None
    right_pos = None
    if result['left_visible']:
        left_x_coords = result['left_landmarks_xyz'][0]
        left_pos = min(left_x_coords), max(left_x_coords)
    if result['right_visible']:
        right_x_coords = result['right_landmarks_xyz'][0]
        right_pos = min(right_x_coords), max(right_x_coords)

    return left_pos, right_pos


def closest_hand(midi_pitch, left_pos, right_pos):
    # Get the outline of the midi pitch and compare the x position of the hands with the outline
    # Return which one is closer to the outline?
    if left_pos is None and right_pos is None:
        return None
    elif left_pos is None:
        return 'right'
    elif right_pos is None:
        return 'left'

    outline = draw_keys_3d.pixel_coordinates_of_key(midi_pitch)
    key_mean_x = np.mean(outline[:, :, 0])
    left_x = left_pos[1] * track_hands.image_width_px
    right_x = right_pos[0] * track_hands.image_width_px

    left_dist = abs(left_x - key_mean_x)
    right_dist = abs(right_x - key_mean_x)
    if left_dist < right_dist:
        return 'left'
    else:
        return 'right'


midi_events = parse_midi_mgs("recording/midi/midi_msg.txt")
for event in midi_events:
    print(f"{event['timestamp']}: {event['message']}")

video_path = "recording/video"
video_events = parse_video(video_path)
for event in video_events:
    print(f"{event['timestamp']}: {event['filename']}")


# Combine and sort events by timestamp
all_events = midi_events + video_events
all_events.sort(key=lambda event: event['timestamp'])


start_real = time.time()
start_recording = all_events[0]['timestamp']

# Output events in chronological order
img = None
current_notes = []
left_hand, right_hand = None, None  # positions of the hands (min_x, max_x)
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
        msg = event['message']
        if msg.type == 'note_on' and msg.velocity > 0:
            current_notes.append(msg.note)
            hand = closest_hand(msg.note, left_hand, right_hand)
            print("Closest hand:", hand)
            if hand == 'left_hand':
                hand = 'left'
            if hand == 'right_hand':
                hand = 'right'
            osc_client.send_message(
                f"/{hand}/note_on", [msg.note, msg.velocity])
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in current_notes:
                current_notes.remove(msg.note)
    elif event['type'] == 'video':
        print(f"{event['timestamp']}: {event['filename']}")
        img_path = os.path.join(video_path, event['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img, result = track_hands.analyze_frame(img)
            left_hand, right_hand = get_hand_positions(result)
            for midi_pitch in current_notes:
                img = draw_keys_3d.draw_key(img, midi_pitch)

            cv2.imshow('Simulate Recording', img)
            cv2.waitKey(1)

cv2.destroyAllWindows()
