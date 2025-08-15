import json
import os
import time

import cv2
import mido
from pythonosc.udp_client import SimpleUDPClient
import numpy as np
from shapely.geometry import Point, Polygon

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


def point_distance_to_quad(point, quad):
    """
    point: tuple (x, y)
    quad: numpy array mit shape (4, 1, 2)
    Returns 0, if the point is inside the polygon,
    otherwise the minimum distance to the polygon.
    """
    # quad in (4,2) umwandeln
    quad_points = quad[:, 0, :]
    polygon = Polygon(quad_points)
    pt = Point(point)
    if polygon.contains(pt):
        return 0.0
    else:
        return pt.distance(polygon)


def closest_hand_and_finger(midi_pitch, mp_result):
    # Get the outline of the midi pitch and compare the x position of the hands with the outline
    # Return which one is closer to the outline?
    if mp_result['left_visible']:
        left_x_coords = mp_result['left_landmarks_xyz'][0]
        left_x = max(left_x_coords) * track_hands.image_width_px
    if mp_result['right_visible']:
        right_x_coords = mp_result['right_landmarks_xyz'][0]
        right_x = min(right_x_coords) * track_hands.image_width_px

    if not mp_result['left_visible'] and not mp_result['right_visible']:
        return None, None
    elif not mp_result['left_visible']:
        result_hand = 'right'
    elif not mp_result['right_visible']:
        result_hand = 'left'
    else:
        outline = draw_keys_3d.pixel_coordinates_of_key(midi_pitch)
        key_mean_x = np.mean(outline[:, :, 0])
        left_dist = abs(left_x - key_mean_x)
        right_dist = abs(right_x - key_mean_x)
        result_hand = 'left' if left_dist < right_dist else 'right'

    if result_hand == 'left':
        landmarks = mp_result['left_landmarks_xyz']
    else:
        landmarks = mp_result['right_landmarks_xyz']
    finger_tips_idx = [track_hands.MP_THUMB_TIP, track_hands.MP_INDEX_FINGER_TIP, track_hands.MP_MIDDLE_FINGER_TIP,
                       track_hands.MP_RING_FINGER_TIP, track_hands.MP_PINKY_TIP]  # Indexes of finger tips in the landmarks
    min_distance = float('inf')
    result_finger = None
    for tip_idx, tip in enumerate(finger_tips_idx):
        x = landmarks[0][tip] * track_hands.image_width_px
        y = landmarks[1][tip] * track_hands.image_height_px
        point = (x, y)
        distance = point_distance_to_quad(point, outline)
        if distance < min_distance:
            min_distance = distance
            result_finger = tip_idx + 1

    return result_hand, result_finger


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
last_mp_result = None
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
            hand, finger = closest_hand_and_finger(msg.note, last_mp_result)
            print(hand.capitalize(), finger)
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
            img, last_mp_result = track_hands.analyze_frame(img)
            for midi_pitch in current_notes:
                img = draw_keys_3d.draw_key(img, midi_pitch)

            cv2.imshow('Simulate Recording', img)
            cv2.waitKey(1)

cv2.destroyAllWindows()
