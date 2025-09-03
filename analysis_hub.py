from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point, Polygon

import track_hands
import draw_keys_3d
import osc_sender
import tip_on_key


def _point_distance_to_quad(point: tuple[int, int], quad: npt.NDArray[Any]) -> float:
    """
    point: tuple (x, y)
    quad: numpy array with shape (4, 1, 2)
    Returns a negative value (= how much it is inside), if the point is inside the polygon, otherwise the minimum 
    distance to the polygon.
    """
    # Convert quad to shape (4,2)
    quad_points = quad[:, 0, :]
    polygon = Polygon(quad_points)
    pt = Point(point)
    if polygon.contains(pt):
        # Return negative value: how far the point is inside
        # Calculate the distance to the nearest edge and return it as negative
        nearest_point = polygon.boundary.interpolate(polygon.boundary.project(pt))
        return -pt.distance(nearest_point)
    else:
        return pt.distance(polygon)


class AnalysisHub:
    def __init__(self):
        self.last_midi_result = {}
        self.current_notes = {}
        self.last_mp_result = {}
        self.last_image_output: Optional[npt.NDArray[Any]] = None

    def closest_hand_and_fingers(self, midi_pitch: int) -> tuple[str, list[int]]:
        """Find the closest hand and fingers to the given MIDI pitch.

        Args:
            midi_pitch (int): The MIDI pitch to check against.

        Returns:
            tuple: (hand (str), fingers (list of int)). 'hand' is either "left" or "right", and 'fingers' is a list of 
                finger indices (1=thumb, 2=index, ..., 5=pinky) closest to or inside
        """

        outline = draw_keys_3d.pixel_coordinates_of_key(midi_pitch)

        # Get the outline of the midi pitch and compare the x position of the hands with the outline
        # Return which one is closer to the outline
        left_x, right_x = None, None
        if self.last_mp_result["left_visible"]:
            left_x_coords = self.last_mp_result["left_landmarks_xyz"][0]
            left_x = max(left_x_coords) * track_hands.image_width_px
        if self.last_mp_result["right_visible"]:
            right_x_coords = self.last_mp_result["right_landmarks_xyz"][0]
            right_x = min(right_x_coords) * track_hands.image_width_px

        if not self.last_mp_result["left_visible"] and not self.last_mp_result["right_visible"]:
            return "", []
        elif not self.last_mp_result["left_visible"]:
            result_hand = "right"
        elif not self.last_mp_result["right_visible"]:
            result_hand = "left"
        else:
            assert left_x is not None and right_x is not None
            key_mean_x = np.mean(outline[:, :, 0])
            left_dist = abs(left_x - key_mean_x)
            right_dist = abs(right_x - key_mean_x)
            result_hand = "left" if left_dist < right_dist else "right"

        if result_hand == "left":
            landmarks = self.last_mp_result["left_landmarks_xyz"]
        else:
            landmarks = self.last_mp_result["right_landmarks_xyz"]
        finger_tips_idx = [
            track_hands.MP_THUMB_TIP,
            track_hands.MP_INDEX_FINGER_TIP,
            track_hands.MP_MIDDLE_FINGER_TIP,
            track_hands.MP_RING_FINGER_TIP,
            track_hands.MP_PINKY_TIP,
        ]  # Indexes of finger tips in the landmarks
        result_fingers = []
        closest_finger = None
        min_distance = float("inf")

        for tip_idx, tip in enumerate(finger_tips_idx):
            x = landmarks[0][tip] * track_hands.image_width_px
            y = landmarks[1][tip] * track_hands.image_height_px
            point = (x, y)
            distance = _point_distance_to_quad(point, outline)

            # Keep track of the closest finger regardless of whether it's inside or outside
            if distance < min_distance:
                min_distance = distance
                closest_finger = tip_idx + 1

            # If the finger is inside the key area (distance < 0), add it to the list
            if distance < 0:
                result_fingers.append(tip_idx + 1)

        # If no fingers are inside the key area, use the closest one
        if not result_fingers and closest_finger is not None:
            result_fingers = [closest_finger]

        return result_hand, result_fingers

    def process_midi_event(self, event: dict):
        msg = event["message"]
        if msg.type == "note_on" and msg.velocity > 0:
            hand, finger = self.closest_hand_and_fingers(msg.note)
            note_properties = {"velocity": msg.velocity, "hand": hand, "finger": finger}
            self.current_notes[msg.note] = note_properties
            self.last_midi_result = dict(note_properties)
            self.last_midi_result["msg.type"] = "note_on"
            self.last_midi_result["note"] = msg.note
            osc_sender.send_message(f"/{hand}/note_on", msg.note, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in self.current_notes:
                del self.current_notes[msg.note]
            self.last_midi_result = {
                "msg.type": "note_off",
                "note": msg.note,
                "velocity": msg.velocity,
            }

    def process_frame(self, img: npt.NDArray[Any]):
        self.last_image_output = img.copy()
        self.last_mp_result = track_hands.analyze_frame(img_input=img, img_output=self.last_image_output)
        for pitch in self.current_notes.keys():
            tip_on_key.find_tip_on_key(
                pitch,
                self.current_notes[pitch],
                self.last_mp_result,
                img_output=self.last_image_output,
            )


hub = AnalysisHub()
