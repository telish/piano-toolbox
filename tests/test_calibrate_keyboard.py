import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
import cv2
from typing import cast, List, Dict, Any, Tuple

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import calibrate_keyboard
from draw_keys_3d import CorrespondingPoints
import utils
import draw_keys_3d
import keyboard_geometry


@pytest.fixture
def mock_image():
    """Create a mock image for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_keyboard_geometry_file(tmp_path):
    """Create a mock keyboard_geometry.json file with test data."""
    keyboard_dir = tmp_path / "calibration" / "keyboard"
    keyboard_dir.mkdir(parents=True)

    # Create a test keyboard_geometry.json file
    geom_file = keyboard_dir / "keyboard_geometry.json"
    geom_data = {
        "black_height": 95.0,
        "keypoint_mappings": [
            {"pixel": [100, 100], "object": [0.0, 0.0]},
            {"pixel": [500, 100], "object": [keyboard_geometry.KEYBOARD_WIDTH, 0.0]},
            {"pixel": [100, 400], "object": [0.0, keyboard_geometry.WHITE_HEIGHT]},
            {
                "pixel": [500, 400],
                "object": [
                    keyboard_geometry.KEYBOARD_WIDTH,
                    keyboard_geometry.WHITE_HEIGHT,
                ],
            },
        ],
    }
    geom_file.write_text(json.dumps(geom_data))

    # Create a mock keyboard image file
    image_file = keyboard_dir / "foto.png"
    # Create a small test image
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.imwrite(str(image_file), test_image)

    # Return the path to the directory
    return str(tmp_path)


def test_parse_args():
    """Test the argument parser function."""
    # Test with no arguments (should default to live mode with index 0)
    with patch("sys.argv", ["calibrate_keyboard.py"]):
        args = calibrate_keyboard.parse_args()
        assert args.recording is None
        assert args.live == 0

    # Test with recording argument
    with patch("sys.argv", ["calibrate_keyboard.py", "--recording", "test_recording"]):
        args = calibrate_keyboard.parse_args()
        assert args.recording == "test_recording"
        assert args.live is None

    # Test with live argument
    with patch("sys.argv", ["calibrate_keyboard.py", "--live", "1"]):
        args = calibrate_keyboard.parse_args()
        assert args.recording is None
        assert args.live == 1


def test_draw_trapezoid(mock_image):
    """Test the draw_trapezoid function."""
    # Test with fewer than 2 points
    points = [{"pixel": (100, 100), "object": (0.0, 0.0)}]
    calibrate_keyboard.draw_trapezoid(
        mock_image, cast(List[CorrespondingPoints], points)
    )

    # Test with 2-3 points - Note that the draw_trapezoid function actually draws using cv2.polylines
    # which we'll need to mock to verify it's called, rather than checking the image content
    points = [
        {"pixel": (100, 100), "object": (0.0, 0.0)},
        {"pixel": (200, 100), "object": (100.0, 0.0)},
        {"pixel": (200, 200), "object": (100.0, 100.0)},
    ]

    with patch("cv2.polylines") as mock_polylines:
        calibrate_keyboard.draw_trapezoid(
            mock_image, cast(List[CorrespondingPoints], points)
        )
        mock_polylines.assert_called_once()

    # Test with exactly 4 points
    mock_image_2 = np.zeros((480, 640, 3), dtype=np.uint8)
    points = [
        {"pixel": (100, 100), "object": (0.0, 0.0)},
        {"pixel": (300, 100), "object": (keyboard_geometry.KEYBOARD_WIDTH, 0.0)},
        {
            "pixel": (300, 300),
            "object": (
                keyboard_geometry.KEYBOARD_WIDTH,
                keyboard_geometry.WHITE_HEIGHT,
            ),
        },
        {"pixel": (100, 300), "object": (0.0, keyboard_geometry.WHITE_HEIGHT)},
    ]

    with patch(
        "calibrate_keyboard.get_correspondences_without_projection", return_value=points
    ), patch("cv2.polylines") as mock_polylines:
        calibrate_keyboard.draw_trapezoid(
            mock_image_2, cast(List[CorrespondingPoints], points)
        )
        mock_polylines.assert_called_once()


def test_find_closest_point_index():
    """Test the find_closest_point_index function."""
    points = [(100, 100), (200, 200), (300, 300)]

    # Test finding a close point
    idx = calibrate_keyboard.find_closest_point_index(101, 101, points)
    assert idx == 0

    # Test with a point that's too far
    idx = calibrate_keyboard.find_closest_point_index(150, 150, points, max_distance=10)
    assert idx == -1

    # Test with empty points list
    idx = calibrate_keyboard.find_closest_point_index(100, 100, [])
    assert idx == -1


def test_mouse_callback(mock_image):
    """Test the mouse callback function."""
    # Reset global variables
    calibrate_keyboard._state["user_defined_points"] = []
    calibrate_keyboard._state["dragging_index"] = -1

    # Test adding a new point
    calibrate_keyboard.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
    assert len(calibrate_keyboard._state["user_defined_points"]) == 1
    assert calibrate_keyboard._state["user_defined_points"][0]["pixel"] == (100, 100)
    assert calibrate_keyboard._state["dragging_index"] == 0

    # Test dragging a point
    calibrate_keyboard.mouse_callback(cv2.EVENT_MOUSEMOVE, 150, 150, 0, None)
    assert calibrate_keyboard._state["user_defined_points"][0]["pixel"] == (150, 150)

    # Test releasing the mouse button
    calibrate_keyboard.mouse_callback(cv2.EVENT_LBUTTONUP, 150, 150, 0, None)
    assert calibrate_keyboard._state["dragging_index"] == -1

    # Test clicking near an existing point
    with patch("calibrate_keyboard.find_closest_point_index", return_value=0):
        calibrate_keyboard.mouse_callback(cv2.EVENT_LBUTTONDOWN, 151, 151, 0, None)
        assert (
            len(calibrate_keyboard._state["user_defined_points"]) == 1
        )  # No new point added
        assert (
            calibrate_keyboard._state["dragging_index"] == 0
        )  # Dragging the existing point


def test_save_coords(mock_image, mock_keyboard_geometry_file):
    """Test saving coordinates to file."""
    # Set up test points
    calibrate_keyboard._state["user_defined_points"] = [
        {"pixel": (100, 100), "object": (0.0, 0.0)},
        {"pixel": (300, 100), "object": (keyboard_geometry.KEYBOARD_WIDTH, 0.0)},
        {
            "pixel": (300, 300),
            "object": (
                keyboard_geometry.KEYBOARD_WIDTH,
                keyboard_geometry.WHITE_HEIGHT,
            ),
        },
        {"pixel": (100, 300), "object": (0.0, keyboard_geometry.WHITE_HEIGHT)},
    ]

    # Mock file operations
    with patch(
        "utils.get_keyboard_geometry_file_path",
        return_value=os.path.join(
            mock_keyboard_geometry_file,
            "calibration",
            "keyboard",
            "keyboard_geometry.json",
        ),
    ), patch(
        "utils.get_keyboard_image_file_path",
        return_value=os.path.join(
            mock_keyboard_geometry_file, "calibration", "keyboard", "foto.png"
        ),
    ), patch(
        "json.dump"
    ) as mock_json_dump, patch(
        "cv2.imwrite"
    ) as mock_imwrite:

        calibrate_keyboard.save_coords(mock_image)

        # Check that json.dump and cv2.imwrite were called
        mock_json_dump.assert_called_once()
        mock_imwrite.assert_called_once()


def test_add_object_coords():
    """Test adding object coordinates to points."""
    # Set up test points with missing object coordinates
    points = [
        {"pixel": (100, 100), "object": None},
        {
            "pixel": (200, 200),
            "object": (100.0, 100.0),
        },  # This one already has coordinates
    ]

    # Mock find_closest_point
    with patch("calibrate_keyboard.find_closest_point", return_value=(0.0, 0.0)):
        calibrate_keyboard.add_object_coords(cast(List[CorrespondingPoints], points))

        # Check that object coordinates were added to the first point
        assert points[0]["object"] == (0.0, 0.0)
        # Check that the second point was not changed
        assert points[1]["object"] == (100.0, 100.0)


def test_find_closest_point():
    """Test finding the closest point on the keyboard."""
    test_point = {"pixel": (150, 150)}

    # Mock necessary functions
    with patch(
        "draw_keys_3d.pixel_coordinates_of_key",
        return_value=np.array(
            [[[100, 100]], [[100, 200]], [[200, 200]], [[200, 100]]], dtype=np.float32
        ),
    ), patch(
        "keyboard_geometry.key_points",
        return_value=[(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)],
    ):

        # Call find_closest_point
        result = calibrate_keyboard.find_closest_point(test_point)

        # Check that it returns valid coordinates
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, (int, float)) for x in result)


def test_get_correspondences_without_projection():
    """Test organizing 4 points into a correct trapezoid."""
    # Create 4 test points in random order
    points = [
        {"pixel": (300, 300), "object": None},  # bottom-right
        {"pixel": (100, 100), "object": None},  # top-left
        {"pixel": (300, 100), "object": None},  # top-right
        {"pixel": (100, 300), "object": None},  # bottom-left
    ]

    # Call the function
    result = calibrate_keyboard.get_correspondences_without_projection(
        cast(List[CorrespondingPoints], points)
    )

    # Check that points were correctly ordered and assigned object coordinates
    assert len(result) == 4

    # Check top-left
    assert result[0]["pixel"] == (100, 100)
    assert result[0]["object"] == keyboard_geometry.KEYBOARD_OUTLINE["top-left"]

    # Check top-right
    assert result[1]["pixel"] == (300, 100)
    assert result[1]["object"] == keyboard_geometry.KEYBOARD_OUTLINE["top-right"]

    # Check bottom-left (note the order in the return value)
    assert result[2]["pixel"] == (100, 300)
    assert result[2]["object"] == keyboard_geometry.KEYBOARD_OUTLINE["bottom-left"]

    # Check bottom-right
    assert result[3]["pixel"] == (300, 300)
    assert result[3]["object"] == keyboard_geometry.KEYBOARD_OUTLINE["bottom-right"]


@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.waitKey")
@patch("cv2.destroyAllWindows")
def test_main_recording_mode(
    mock_destroy,
    mock_waitkey,
    mock_imshow,
    mock_videocapture,
    mock_keyboard_geometry_file,
):
    """Test main function in recording mode."""
    # Setup mock video capture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_videocapture.return_value = mock_cap

    # Setup mock waitKey to quit after one iteration
    mock_waitkey.return_value = ord("q")

    # Create recording path
    recording_path = os.path.join(mock_keyboard_geometry_file, "recording")
    os.makedirs(os.path.join(recording_path, "video"), exist_ok=True)

    # Patch command line arguments and other functions
    with patch(
        "calibrate_keyboard.parse_args",
        return_value=MagicMock(recording=recording_path, live=None),
    ), patch(
        "utils.flip_image", return_value=np.zeros((480, 640, 3), dtype=np.uint8)
    ), patch(
        "calibrate_keyboard.save_coords"
    ), patch(
        "os.path.abspath", return_value=recording_path
    ):

        # Run main function
        calibrate_keyboard.main()

        # Check that windows were created and destroyed
        mock_destroy.assert_called_once()
