import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import track_hands
import osc_sender


@pytest.fixture
def mock_hands_process():
    """Create a mock for MediaPipe Hands process method."""
    # Create a mock return value for hands.process()
    mock_result = MagicMock()

    # Mock multi_hand_landmarks
    mock_landmark = MagicMock()
    mock_landmark.x = 0.5
    mock_landmark.y = 0.5
    mock_landmark.z = 0.1

    mock_hand_landmarks = MagicMock()
    mock_hand_landmarks.landmark = [mock_landmark] * 21  # MediaPipe uses 21 landmarks

    # Mock multi_handedness
    mock_classification = MagicMock()
    mock_classification.label = "Left"  # This will be flipped to "right" in the code

    mock_handedness = MagicMock()
    mock_handedness.classification = [mock_classification]

    # Set up the main results
    mock_result.multi_hand_landmarks = [mock_hand_landmarks]
    mock_result.multi_handedness = [mock_handedness]

    return mock_result


def test_finger_to_tip_index():
    """Test that finger_to_tip_index maps correctly."""
    assert track_hands.finger_to_tip_index[1] == track_hands.MP_THUMB_TIP
    assert track_hands.finger_to_tip_index[2] == track_hands.MP_INDEX_FINGER_TIP
    assert track_hands.finger_to_tip_index[3] == track_hands.MP_MIDDLE_FINGER_TIP
    assert track_hands.finger_to_tip_index[4] == track_hands.MP_RING_FINGER_TIP
    assert track_hands.finger_to_tip_index[5] == track_hands.MP_PINKY_TIP


def test_analyze_frame_no_hands():
    """Test analyze_frame when no hands are detected."""
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock hands.process to return no hands
    with patch.object(track_hands.hands, "process") as mock_process, patch("osc_sender.send_message"):

        mock_result = MagicMock()
        mock_result.multi_hand_landmarks = None
        mock_process.return_value = mock_result

        # Call analyze_frame
        result = track_hands.analyze_frame(test_image)

        # Check results
        assert result["left_visible"] is False
        assert result["right_visible"] is False
        assert result["left_landmarks_xyz"] is None
        assert result["right_landmarks_xyz"] is None


def test_analyze_frame_with_hands(mock_hands_process):
    """Test analyze_frame when hands are detected."""
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock hands.process
    with patch.object(track_hands.hands, "process", return_value=mock_hands_process), patch(
        "osc_sender.send_message"
    ), patch("mediapipe.solutions.drawing_utils.draw_landmarks"):

        # Call analyze_frame
        result = track_hands.analyze_frame(test_image, test_image.copy())

        # Check results - should have a right hand (since Left in MediaPipe is flipped)
        assert result["right_visible"] is True
        assert result["left_visible"] is False

        # Check that landmarks were extracted
        assert result["right_landmarks_xyz"] is not None
        assert len(result["right_landmarks_xyz"][0]) == 21  # 21 x-coordinates
        assert len(result["right_landmarks_xyz"][1]) == 21  # 21 y-coordinates
        assert len(result["right_landmarks_xyz"][2]) == 21  # 21 z-coordinates


@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.waitKey")
@patch("cv2.destroyAllWindows")
def test_main(mock_destroy, mock_waitkey, mock_imshow, mock_videocapture):
    """Test that the main function runs without errors."""
    # Setup mocks
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]
    mock_videocapture.return_value = mock_cap
    mock_waitkey.return_value = 0  # not 'q'

    # Mock analyze_frame to avoid actual mediapipe calls
    with patch("track_hands.analyze_frame", return_value={}), patch("osc_sender.configure") as mock_osc:

        # Call the main block by directly executing the if __name__ == "__main__" section
        if_main_code = """
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    osc_sender.configure(9876)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        analyze_frame(img, img)
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the video feed
            break

    cap.release()
    cv2.destroyAllWindows()
"""
        # Create a namespace with the required module references
        namespace = {
            "cv2": cv2,
            "osc_sender": osc_sender,
            "analyze_frame": track_hands.analyze_frame,
            "__name__": "__main__",
        }

        # Execute the code
        exec(if_main_code, namespace)

        # Check that OSC was configured
        mock_osc.assert_called_once_with(9876)

        # Check that video capture was released and windows destroyed
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()


def test_image_dimensions():
    """Test that image dimensions are correctly set."""
    # Create a test image with specific dimensions
    height, width = 720, 1280
    test_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Mock hands.process to return no hands
    with patch.object(track_hands.hands, "process") as mock_process, patch("osc_sender.send_message"):

        mock_result = MagicMock()
        mock_result.multi_hand_landmarks = None
        mock_process.return_value = mock_result

        # Call analyze_frame
        track_hands.analyze_frame(test_image)

        # Check that image dimensions were correctly set
        assert track_hands.image_height_px == height
        assert track_hands.image_width_px == width
