"""
Analysis hub provides the main entrance point for all analyses and coordinates results between
analysis modules.
"""

import mido
import numpy as np
from shapely.geometry import Point, Polygon

import draw_keys_3d
import osc_sender
import tip_on_key
import track_hands
from datatypes import HandLiteral, Image, MidiResult, TrackingResult


class ProcessingHub:
    """Coordinates analysis results between modules."""

    def __init__(self) -> None:
        self.last_midi_result: MidiResult | None = None
        self.current_notes: dict[int, MidiResult] = {}
        self.last_mp_result: TrackingResult = TrackingResult()
        self.last_image_output: Image | None = None

    def _closest_hand_and_fingers(self, midi_pitch: int) -> tuple[HandLiteral, list[int]]:
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
        left_x, right_x = float("-inf"), float("inf")
        if self.last_mp_result.left_visible:
            left_x_coords = self.last_mp_result.left_landmarks_xyz[0]
            left_x = max(left_x_coords) * track_hands.image_width_px
        if self.last_mp_result.right_visible:
            right_x_coords = self.last_mp_result.right_landmarks_xyz[0]
            right_x = min(right_x_coords) * track_hands.image_width_px

        if not self.last_mp_result.left_visible and not self.last_mp_result.right_visible:
            return "", []
        elif not self.last_mp_result.left_visible:
            result_hand = "right"
        elif not self.last_mp_result.right_visible:
            result_hand = "left"
        else:
            assert left_x is not None and right_x is not None
            key_mean_x = np.mean(outline[:, :, 0])
            left_dist = abs(left_x - key_mean_x)
            right_dist = abs(right_x - key_mean_x)
            result_hand = "left" if left_dist < right_dist else "right"

        if result_hand == "left":
            landmarks = self.last_mp_result.left_landmarks_xyz
        else:
            landmarks = self.last_mp_result.right_landmarks_xyz
        finger_tips_idx = [
            track_hands.MP_THUMB_TIP,
            track_hands.MP_INDEX_FINGER_TIP,
            track_hands.MP_MIDDLE_FINGER_TIP,
            track_hands.MP_RING_FINGER_TIP,
            track_hands.MP_PINKY_TIP,
        ]  # Indexes of finger tips in the landmarks
        result_fingers: list[int] = []
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

    def draw_results(self, img: Image) -> None:
        """Draw analysis results on the image."""
        # Draw notes
        for midi_pitch, note_props in self.current_notes.items():
            if note_props["hand"] == "left":
                color = (0, 0, 200)  # Red color
            elif note_props["hand"] == "right":
                color = (0, 200, 0)  # Green color
            else:
                color = (200, 200, 0)  # Yellow for unknown hand

            annotation = f"{', '.join(str(x) for x in note_props['fingers'])}"
            img = draw_keys_3d.draw_key(img, midi_pitch, color, annotation)

        draw_keys_3d.draw_keyboard(img, (0, 165, 255), outline_only=True)  # Orange color in BGR format

    def process_midi_event(self, timestamp: float, msg: mido.Message) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            hand, fingers = self._closest_hand_and_fingers(msg.note)

            midi_result: MidiResult = {
                "type": "note_on",
                "pitch": msg.note,
                "velocity": msg.velocity,
                "hand": hand,
                "fingers": fingers,
            }
            self.current_notes[msg.note] = midi_result
            self.last_midi_result = dict(midi_result)
            self.send_note_on_osc(msg.note, msg.velocity, hand, fingers)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            fingers = []
            hand = ""
            if msg.note in self.current_notes:
                old_result = self.current_notes[msg.note]
                fingers = old_result["fingers"]
                hand = old_result["hand"]
                del self.current_notes[msg.note]
            self.last_midi_result = {
                "type": "note_off",
                "pitch": msg.note,
                "velocity": msg.velocity,
                "hand": hand,
                "fingers": fingers,
            }
            self.send_note_off_osc(msg.note, hand, fingers)

    def send_note_on_osc(self, midi_pitch: int, velocity: int, hand: str, fingers: list[int]) -> None:
        osc_sender.send_message(f"/note", midi_pitch, velocity)
        osc_sender.send_message(f"/{hand}/note", midi_pitch, velocity)
        for finger in fingers:
            osc_sender.send_message(f"/{hand}/{finger}/note", midi_pitch, velocity)

    def send_note_off_osc(self, midi_pitch: int, hand: str, fingers: list[int]) -> None:
        velocity = 0
        osc_sender.send_message(f"/note", midi_pitch, velocity)
        osc_sender.send_message(f"/{hand}/note", midi_pitch, velocity)
        for finger in fingers:
            osc_sender.send_message(f"/{hand}/{finger}/note", midi_pitch, velocity)

    def process_frame(self, timestamp: float, img: Image) -> None:
        self.last_image_output = img.copy()
        self.last_mp_result = track_hands.analyze_frame(img_input=img, img_output=self.last_image_output)
        assert self.last_mp_result is not None
        for pitch, note_properties in self.current_notes.items():
            tip_on_key.find_tip_on_key(
                pitch,
                note_properties,
                self.last_mp_result,
                img_output=self.last_image_output,
            )


def _point_distance_to_quad(point: tuple[float, float], quad: Image) -> float:
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


hub = ProcessingHub()
