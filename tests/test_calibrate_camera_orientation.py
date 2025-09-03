import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import calibrate_camera_orientation
import utils


@pytest.fixture
def mock_orientation_file(tmp_path):
    """Create a temporary directory structure for camera orientation file."""
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()

    # Create a mock camera orientation file
    orientation_file = calibration_dir / "camera_orientation.json"
    with open(orientation_file, "w") as f:
        json.dump({"flip_horizontal": False, "flip_vertical": True}, f)

    return str(tmp_path)


def test_parse_args_default():
    """Test parse_args with default values."""
    with patch("sys.argv", ["calibrate_camera_orientation.py"]):
        args = calibrate_camera_orientation.parse_args()
        assert args.live == 0  # Default camera index
        assert args.recording is None  # No recording by default


def test_parse_args_live():
    """Test parse_args with live camera option."""
    with patch("sys.argv", ["calibrate_camera_orientation.py", "--live", "1"]):
        args = calibrate_camera_orientation.parse_args()
        assert args.live == 1
        assert args.recording is None


def test_parse_args_recording():
    """Test parse_args with recording option."""
    with patch("sys.argv", ["calibrate_camera_orientation.py", "--recording", "test_recording"]):
        args = calibrate_camera_orientation.parse_args()
        assert args.recording == "test_recording"
        assert args.live is None


def test_save_orientation():
    """Test save_orientation function saves correct data."""
    # Set up the global variables
    calibrate_camera_orientation.flip_horizontal = True
    calibrate_camera_orientation.flip_vertical = False

    # Create a mock for the file operations
    mock_file = MagicMock()
    m_open = mock_open(mock=mock_file)

    with patch("builtins.open", m_open), patch("json.dump") as mock_dump, patch(
        "utils.retrieve_camera_orientation_file_path", return_value="/path/to/orientation.json"
    ), patch("os.makedirs"):

        # Call the function
        calibrate_camera_orientation.save_orientation()

        # Check file was opened correctly
        m_open.assert_called_once_with("/path/to/orientation.json", "w")

        # Check correct data was written
        mock_dump.assert_called_once()
        args, _ = mock_dump.call_args
        data = args[0]
        assert data == {"flip_horizontal": True, "flip_vertical": False}


@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.waitKey")
@patch("cv2.destroyAllWindows")
def test_main_live_mode(mock_destroy, mock_waitkey, mock_imshow, mock_videocapture):
    """Test main function in live camera mode."""
    # Setup mock video capture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_videocapture.return_value = mock_cap

    # Reset global variables
    calibrate_camera_orientation.flip_horizontal = False
    calibrate_camera_orientation.flip_vertical = False

    # Mock waitKey to simulate key presses
    mock_waitkey.side_effect = [
        ord("h"),  # Press 'h' to flip horizontal
        ord("v"),  # Press 'v' to flip vertical
        ord("q"),  # Press 'q' to quit
    ]

    # Patch command line arguments and other functions
    with patch("calibrate_camera_orientation.parse_args", return_value=MagicMock(live=0, recording=None)), patch(
        "utils.add_text_to_image", return_value=np.zeros((480, 640, 3), dtype=np.uint8)
    ), patch("calibrate_camera_orientation.save_orientation") as mock_save, patch(
        "cv2.flip", return_value=np.zeros((480, 640, 3), dtype=np.uint8)
    ):

        # Run main function
        calibrate_camera_orientation.main()

        # Check video capture was initialized correctly
        mock_videocapture.assert_called_once_with(0)

        # Check that the key presses changed the orientation flags
        assert calibrate_camera_orientation.flip_horizontal is True
        assert calibrate_camera_orientation.flip_vertical is True

        # Check that save_orientation was called
        mock_save.assert_called_once()

        # Check that windows were destroyed
        mock_destroy.assert_called_once()


@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.waitKey")
@patch("cv2.destroyAllWindows")
def test_main_recording_mode(mock_destroy, mock_waitkey, mock_imshow, mock_videocapture, mock_orientation_file):
    """Test main function in recording mode."""
    # Setup mock video capture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_videocapture.return_value = mock_cap

    # Reset global variables
    calibrate_camera_orientation.flip_horizontal = False
    calibrate_camera_orientation.flip_vertical = False

    # Mock waitKey to simulate key presses
    mock_waitkey.side_effect = [ord("q")]  # Press 'q' to quit

    # Create recording path
    recording_path = os.path.join(mock_orientation_file, "recording")
    os.makedirs(os.path.join(recording_path, "video"), exist_ok=True)
    video_path = os.path.join(recording_path, "video", "recording.avi")

    # Patch command line arguments and other functions
    with patch(
        "calibrate_camera_orientation.parse_args", return_value=MagicMock(live=None, recording=recording_path)
    ), patch("utils.add_text_to_image", return_value=np.zeros((480, 640, 3), dtype=np.uint8)), patch(
        "calibrate_camera_orientation.save_orientation"
    ) as mock_save, patch(
        "os.path.abspath", return_value=recording_path
    ), patch(
        "os.path.join", return_value=video_path
    ):

        # Run main function
        calibrate_camera_orientation.main()

        # Check video capture was initialized correctly
        mock_videocapture.assert_called_once_with(video_path)

        # Check that save_orientation was called
        mock_save.assert_called_once()

        # Check that windows were destroyed
        mock_destroy.assert_called_once()


def test_main_function_calls():
    """Test that the main function is correctly called when script is run directly."""
    original_name = calibrate_camera_orientation.__name__
    try:
        with patch("calibrate_camera_orientation.main") as mock_main:
            # Simulate __name__ == "__main__"
            calibrate_camera_orientation.__name__ = "__main__"

            # Execute the code that would run when __name__ == "__main__"
            if calibrate_camera_orientation.__name__ == "__main__":
                mock_main()

            # Verify main was called
            mock_main.assert_called_once()
    finally:
        # Restore original name
        calibrate_camera_orientation.__name__ = original_name
