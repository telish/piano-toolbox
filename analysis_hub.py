import numpy as np
from shapely.geometry import Point, Polygon

import track_hands
import draw_keys_3d
import osc_sender


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


last_midi_result = None
current_notes = {}


def process_midi_event(event):
    global current_notes, last_midi_result
    msg = event['message']
    if msg.type == 'note_on' and msg.velocity > 0:
        hand, finger = closest_hand_and_finger(msg.note, last_mp_result)
        note_properties = {
            "velocity": msg.velocity,
            "hand": hand,
            "finger": finger
        }
        current_notes[msg.note] = note_properties
        last_midi_result = dict(note_properties)
        last_midi_result["msg.type"] = "note_on"
        last_midi_result["note"] = msg.note
        osc_sender.send_message(f"/{hand}/note_on", msg.note, msg.velocity)
    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
        if msg.note in current_notes:
            del current_notes[msg.note]
        last_midi_result = {
            "msg.type": "note_off",
            "note": msg.note,
            "velocity": msg.velocity
        }


last_mp_result = None
last_mp_image = None


def process_frame(img):
    global last_mp_result, last_mp_image
    last_mp_image, last_mp_result = track_hands.analyze_frame(img)
