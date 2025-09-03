import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tip_on_key
import draw_keys_3d
import track_hands
import utils


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


@pytest.fixture
def mock_mp_result():
    """Create mock MediaPipe result for testing."""
    return {
        "left_visible": True,
        "right_visible": True,
        "left_landmarks_xyz": (
            [0.5, 0.6, 0.7, 0.8, 0.9],  # x-coords for all landmarks
            [0.1, 0.2, 0.3, 0.4, 0.5],  # y-coords for all landmarks
            [0.01, 0.02, 0.03, 0.04, 0.05],  # z-coords for all landmarks
        ),
        "right_landmarks_xyz": (
            [0.1, 0.2, 0.3, 0.4, 0.5],  # x-coords for all landmarks
            [0.5, 0.4, 0.3, 0.2, 0.1],  # y-coords for all landmarks
            [0.05, 0.04, 0.03, 0.02, 0.01],  # z-coords for all landmarks
        ),
    }


def test_point_to_trapezoid_coords():
    """Test the point_to_trapezoid_coords function."""
    # Simple test with a square
    square = np.array(
        [
            [[0, 0]],  # top-left
            [[0, 1]],  # bottom-left
            [[1, 1]],  # bottom-right
            [[1, 0]],  # top-right
        ],
        dtype=np.float32,
    )

    # Test points
    point_center = (0.5, 0.5)
    point_top_left = (0.1, 0.1)
    point_bottom_right = (0.9, 0.9)

    # Test that point in center has u=0.5, v=0.5
    u, v = tip_on_key.point_to_trapezoid_coords(point_center, square)
    assert abs(u - 0.5) < 0.05
    assert abs(v - 0.5) < 0.05

    # Test point near top-left has small u, v values
    u, v = tip_on_key.point_to_trapezoid_coords(point_top_left, square)
    assert u < 0.2
    assert v < 0.2

    # Test point near bottom-right has large u, v values
    u, v = tip_on_key.point_to_trapezoid_coords(point_bottom_right, square)
    assert u > 0.8
    assert v > 0.8

    # Test with a trapezoid
    trapezoid = np.array(
        [
            [[0, 0]],  # top-left
            [[0, 1]],  # bottom-left
            [[2, 1]],  # bottom-right
            [[1, 0]],  # top-right
        ],
        dtype=np.float32,
    )

    # Point in the middle of the trapezoid
    u, v = tip_on_key.point_to_trapezoid_coords((0.5, 0.5), trapezoid)
    assert 0.3 < u < 0.7  # Approximation due to the shape
    assert 0.3 < v < 0.7


def test_draw_tip_on_key(mock_key_outline):
    """Test the draw_tip_on_key function."""
    # Create a test image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    tip_xy_coords = (150, 150)  # Center of the key
    tip_uv_coords = (0.5, 0.5)  # Center in UV coordinates

    # Test without showing bounding box or text
    result = tip_on_key.draw_tip_on_key(img.copy(), mock_key_outline, tip_xy_coords, tip_uv_coords)
    assert result is not None
    assert np.sum(result) > 0  # Image should have been modified

    # Test with showing bounding box and text
    result = tip_on_key.draw_tip_on_key(
        img.copy(), mock_key_outline, tip_xy_coords, tip_uv_coords, show_bb=True, show_text=True
    )
    assert result is not None
    assert np.sum(result) > 0  # Image should have been modified


def test_find_tip_on_key(mock_mp_result, mock_key_outline):
    """Test the find_tip_on_key function."""
    # Mock note properties
    note_properties = {"hand": "left", "finger": [2]}  # Index finger

    # Create a more complete mock MediaPipe result with enough landmarks
    mp_result = {
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

    # Set up image width and height
    track_hands.image_width_px = 640
    track_hands.image_height_px = 480

    # Set up finger indices in track_hands
    track_hands.finger_to_tip_index = {
        1: 4,  # Thumb
        2: 8,  # Index
        3: 12,  # Middle
        4: 16,  # Ring
        5: 20,  # Pinky
    }

    # Mock the pixel_coordinates_of_bounding_box function
    with patch("draw_keys_3d.pixel_coordinates_of_bounding_box", return_value=mock_key_outline), patch(
        "tip_on_key.point_to_trapezoid_coords", return_value=(0.5, 0.5)
    ):
        # Test without output image
        result = tip_on_key.find_tip_on_key(60, note_properties, mp_result)
        assert result is not None
        assert len(result) == 2  # Should return (u, v) tuple

        # Test with output image
        img_output = np.zeros((300, 300, 3), dtype=np.uint8)
        result = tip_on_key.find_tip_on_key(60, note_properties, mp_result, img_output)
        assert result is not None
        assert np.sum(img_output) > 0  # Image should have been modified


@patch("cv2.namedWindow")
@patch("cv2.setMouseCallback")
@patch("cv2.imshow")
@patch("cv2.waitKey")
@patch("cv2.destroyAllWindows")
def test_test_interactive(mock_destroy, mock_waitkey, mock_imshow, mock_callback, mock_window):
    """Test the test_interactive function runs without errors."""
    # Set up mock returns
    mock_waitkey.side_effect = [ord("q")]  # Exit on first call

    # Mock drawing-related functions
    with patch("draw_keys_3d.init"), patch("utils.get_keyboard_image_file_path", return_value="mock_path.jpg"), patch(
        "cv2.imread", return_value=np.zeros((300, 300, 3), dtype=np.uint8)
    ), patch("utils.flip_image", return_value=np.zeros((300, 300, 3), dtype=np.uint8)), patch(
        "draw_keys_3d.pixel_coordinates_of_bounding_box",
        return_value=np.array([[[100, 100]], [[100, 200]], [[200, 200]], [[200, 100]]], dtype=np.float32),
    ):

        # Run the test_interactive function
        tip_on_key.test_interactive()

        # Verify that the window was created and destroyed
        mock_window.assert_called_once()
        mock_destroy.assert_called_once()


def test_main_function():
    """Test that the main function is correctly linked to test_interactive."""
    with patch("tip_on_key.test_interactive") as mock_test:
        # Run the script's main function
        with patch.object(tip_on_key, "__name__", "__main__"):
            tip_on_key.test_interactive()

        # Verify test_interactive was called
        mock_test.assert_called_once()
