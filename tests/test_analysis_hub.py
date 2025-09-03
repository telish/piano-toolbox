import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis_hub


@pytest.fixture
def mock_mp_result():
    """Create mock MediaPipe result for testing."""
    return {
        "left_visible": True,
        "right_visible": True,
        "left_landmarks_xyz": (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # x-coords for all landmarks
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # y-coords for all landmarks
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],  # z-coords for all landmarks
        ),
        "right_landmarks_xyz": (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # x-coords for all landmarks
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # y-coords for all landmarks
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],  # z-coords for all landmarks
        ),
    }


@pytest.fixture
def mock_key_outline():
    """Create mock key outline for testing."""
    return np.array(
        [
            [[100, 100]],  # top-left
            [[100, 200]],  # bottom-left
            [[200, 200]],  # bottom-right
            [[200, 100]],  # top-right
        ],
        dtype=np.float32,
    )


def test_point_distance_to_quad():
    """Test the _point_distance_to_quad function."""
    # Create a simple square polygon
    quad = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.float32)

    # Test point inside polygon
    distance = analysis_hub._point_distance_to_quad((5, 5), quad)
    assert distance < 0  # Inside points have negative distance

    # Test point outside polygon
    distance = analysis_hub._point_distance_to_quad((15, 15), quad)
    assert distance > 0  # Outside points have positive distance


def test_analysis_hub_init():
    """Test initializing the AnalysisHub class."""
    hub = analysis_hub.AnalysisHub()
    assert hub.last_midi_result == {}
    assert hub.current_notes == {}
    assert hub.last_mp_result == {}
    assert hub.last_image_output is None


def test_closest_hand_and_fingers_no_hands():
    """Test closest_hand_and_fingers when no hands are visible."""
    hub = analysis_hub.AnalysisHub()
    hub.last_mp_result = {"left_visible": False, "right_visible": False}

    # Mock the draw_keys_3d.pixel_coordinates_of_key function
    with patch(
        "draw_keys_3d.pixel_coordinates_of_key",
        return_value=np.array([[[100, 100]], [[100, 200]], [[200, 200]], [[200, 100]]], dtype=np.float32),
    ):
        hand, fingers = hub.closest_hand_and_fingers(60)  # Middle C
        assert hand == ""
        assert fingers == []


def test_closest_hand_and_fingers_left_only():
    """Test closest_hand_and_fingers when only left hand is visible."""
    hub = analysis_hub.AnalysisHub()
    hub.last_mp_result = {
        "left_visible": True,
        "right_visible": False,
        "left_landmarks_xyz": ([0.5] * 21, [0.5] * 21, [0.1] * 21),
    }

    # Set up track_hands module values
    analysis_hub.track_hands.image_width_px = 640
    analysis_hub.track_hands.image_height_px = 480

    # Mock pixel_coordinates_of_key
    with patch(
        "draw_keys_3d.pixel_coordinates_of_key",
        return_value=np.array([[[100, 100]], [[100, 200]], [[200, 200]], [[200, 100]]], dtype=np.float32),
    ):
        hand, fingers = hub.closest_hand_and_fingers(60)  # Middle C
        assert hand == "left"
        assert isinstance(fingers, list)


def test_closest_hand_and_fingers_both_hands(mock_mp_result, mock_key_outline):
    """Test closest_hand_and_fingers when both hands are visible."""
    hub = analysis_hub.AnalysisHub()

    # Create a mock with complete landmarks
    hub.last_mp_result = {
        "left_visible": True,
        "right_visible": True,
        "left_landmarks_xyz": (
            [0.1] * 21,  # x-coords for all landmarks (21 landmarks total)
            [0.2] * 21,  # y-coords for all landmarks
            [0.01] * 21,  # z-coords for all landmarks
        ),
        "right_landmarks_xyz": (
            [0.8] * 21,  # x-coords for all landmarks
            [0.7] * 21,  # y-coords for all landmarks
            [0.02] * 21,  # z-coords for all landmarks
        ),
    }

    # Set up track_hands module values
    analysis_hub.track_hands.image_width_px = 640
    analysis_hub.track_hands.image_height_px = 480

    # Set up finger indices in track_hands
    analysis_hub.track_hands.MP_THUMB_TIP = 4
    analysis_hub.track_hands.MP_INDEX_FINGER_TIP = 8
    analysis_hub.track_hands.MP_MIDDLE_FINGER_TIP = 12
    analysis_hub.track_hands.MP_RING_FINGER_TIP = 16
    analysis_hub.track_hands.MP_PINKY_TIP = 20

    # Mock pixel_coordinates_of_key
    with patch("draw_keys_3d.pixel_coordinates_of_key", return_value=mock_key_outline), patch(
        "analysis_hub._point_distance_to_quad", return_value=0.0
    ):  # No distance, so the fingers are exactly on the key
        hand, fingers = hub.closest_hand_and_fingers(60)  # Middle C
        assert hand in ["left", "right"]
        assert isinstance(fingers, list)


def test_process_midi_event_note_on():
    """Test processing a note_on MIDI event."""
    hub = analysis_hub.AnalysisHub()

    # Create a mock MIDI message
    mock_msg = MagicMock()
    mock_msg.type = "note_on"
    mock_msg.note = 60  # Middle C
    mock_msg.velocity = 64

    # Create a mock event dictionary
    event = {"message": mock_msg}

    # Mock closest_hand_and_fingers to return fixed values
    with patch.object(hub, "closest_hand_and_fingers", return_value=("left", [2])), patch("osc_sender.send_message"):
        hub.process_midi_event(event)

        # Check that the note was added to current_notes
        assert 60 in hub.current_notes
        assert hub.current_notes[60]["velocity"] == 64
        assert hub.current_notes[60]["hand"] == "left"
        assert hub.current_notes[60]["finger"] == [2]

        # Check that last_midi_result was updated
        assert hub.last_midi_result["msg.type"] == "note_on"
        assert hub.last_midi_result["note"] == 60


def test_process_midi_event_note_off():
    """Test processing a note_off MIDI event."""
    hub = analysis_hub.AnalysisHub()

    # Add a note to current_notes
    hub.current_notes[60] = {"velocity": 64, "hand": "left", "finger": [2]}

    # Create a mock MIDI message for note_off
    mock_msg = MagicMock()
    mock_msg.type = "note_off"
    mock_msg.note = 60  # Middle C
    mock_msg.velocity = 0

    # Create a mock event dictionary
    event = {"message": mock_msg}

    hub.process_midi_event(event)

    # Check that the note was removed from current_notes
    assert 60 not in hub.current_notes

    # Check that last_midi_result was updated
    assert hub.last_midi_result["msg.type"] == "note_off"
    assert hub.last_midi_result["note"] == 60


def test_process_frame():
    """Test processing a video frame."""
    hub = analysis_hub.AnalysisHub()

    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add a test note
    hub.current_notes[60] = {"velocity": 64, "hand": "left", "finger": [2]}

    # Mock the dependencies
    with patch("track_hands.analyze_frame", return_value={"left_visible": True}), patch("tip_on_key.find_tip_on_key"):

        hub.process_frame(test_image)

        # Check that last_image_output was created
        assert hub.last_image_output is not None
        assert hub.last_image_output.shape == test_image.shape

        # Check that last_mp_result was updated
        assert hub.last_mp_result == {"left_visible": True}
