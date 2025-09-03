import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import cv2


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keyboard_geometry


@pytest.fixture
def mock_keyboard_geometry_file(tmp_path):
    """Create a mock keyboard_geometry.json file with test data."""
    keyboard_dir = tmp_path / "calibration" / "keyboard"
    keyboard_dir.mkdir(parents=True)

    # Create a test keyboard_geometry.json file
    geom_file = keyboard_dir / "keyboard_geometry.json"
    geom_data = {"black_height": 95.0}
    geom_file.write_text(json.dumps(geom_data))

    # Return the path to the file
    return str(tmp_path)


def test_keyboard_geometry_load_black_height(mock_keyboard_geometry_file):
    """Test that black_height is correctly loaded from the keyboard_geometry.json file."""
    with patch(
        "utils.get_keyboard_geometry_file_path",
        return_value=os.path.join(mock_keyboard_geometry_file, "calibration", "keyboard", "keyboard_geometry.json"),
    ):
        # Reset the module to ensure black_height is reloaded
        black_height = keyboard_geometry.load_black_height()
        assert black_height == 95.0


def test_keyboard_geometry_default_black_height():
    """Test that default black_height is used when file doesn't exist."""
    with patch("utils.get_keyboard_geometry_file_path", return_value="/nonexistent/path"):
        black_height = keyboard_geometry.load_black_height()
        assert black_height == 100.0


def test_keyboard_geometry_pitch_class():
    """Test the pitch_class function."""
    assert keyboard_geometry.pitch_class(60) == "C"  # Middle C
    assert keyboard_geometry.pitch_class(61) == "C#"
    assert keyboard_geometry.pitch_class(62) == "D"
    assert keyboard_geometry.pitch_class(69) == "A"  # A4 (440Hz)
    assert keyboard_geometry.pitch_class(71) == "B"

    # Test octave wrapping
    assert keyboard_geometry.pitch_class(72) == "C"  # C5
    assert keyboard_geometry.pitch_class(84) == "C"  # C6


def test_keyboard_geometry_key_points():
    """Test the key_points function for white and black keys."""
    # Test a white key (middle C = 60)
    white_key_points = keyboard_geometry.key_points(60)
    assert len(white_key_points) == 8  # White keys have 8 points

    # Test a black key (C# = 61)
    black_key_points = keyboard_geometry.key_points(61)
    assert len(black_key_points) == 4  # Black keys have 4 points


def test_keyboard_geometry_key_bounding_box():
    """Test the key_bounding_box function."""
    # Test a white key
    white_box = keyboard_geometry.key_bounding_box(60)  # Middle C
    assert len(white_box) == 4  # Should have 4 corners

    # For a black key, key_bounding_box should return the same as key_points
    black_key = 61  # C#
    black_box = keyboard_geometry.key_bounding_box(black_key)
    black_points = keyboard_geometry.key_points(black_key)
    assert black_box == black_points


def test_keyboard_geometry_re_init():
    """Test the re_init function updates internal data structures."""
    # Save original black_height
    original_black_height = keyboard_geometry.black_height

    try:
        # Change black_height and reinitialize
        keyboard_geometry.black_height = 80.0
        keyboard_geometry.re_init()

        # Test that arrays are still valid after reinitialization
        assert len(keyboard_geometry.left_at_top) == 88
        assert len(keyboard_geometry.left_at_bottom) == 88
        assert len(keyboard_geometry.right_at_top) == 88
        assert len(keyboard_geometry.right_at_bottom) == 88

    finally:
        # Restore original black_height and reinitialize
        keyboard_geometry.black_height = original_black_height
        keyboard_geometry.re_init()


@pytest.mark.parametrize("midi_pitch", [21, 60, 88, 108])  # Test various keys
def test_keyboard_geometry_white_keys(midi_pitch):
    """Test white keys data is consistent."""
    if midi_pitch in keyboard_geometry.white_keys:
        points = keyboard_geometry.key_points(midi_pitch)
        # Check that points form a valid polygon
        assert len(points) == 8
        # Top points should have y=0
        assert points[0][1] == 0
        assert points[7][1] == 0
        # Bottom points should have y=WHITE_HEIGHT
        assert points[3][1] == keyboard_geometry.WHITE_HEIGHT
        assert points[4][1] == keyboard_geometry.WHITE_HEIGHT


@pytest.mark.parametrize("midi_pitch", [22, 61, 73, 106])  # Test various black keys
def test_keyboard_geometry_black_keys(midi_pitch):
    """Test black keys data is consistent."""
    if midi_pitch in keyboard_geometry.black_keys:
        points = keyboard_geometry.key_points(midi_pitch)
        # Check that points form a valid polygon
        assert len(points) == 4
        # Top points should have y=0
        assert points[0][1] == 0
        assert points[3][1] == 0
        # Bottom points should have y=black_height
        assert points[1][1] == keyboard_geometry.black_height
        assert points[2][1] == keyboard_geometry.black_height


@patch("cv2.imshow")
@patch("cv2.waitKey")
def test_keyboard_geometry_main_function(mock_waitkey, mock_imshow):
    """Test the main function runs without errors."""
    # Mock cv2.waitKey to return immediately
    mock_waitkey.return_value = 0

    # Run the main function
    keyboard_geometry.main()

    # Verify that imshow was called
    mock_imshow.assert_called_once()
    assert mock_imshow.call_args[0][0] == "Keyboard Geometry"

    # Verify the image is valid
    img = mock_imshow.call_args[0][1]
    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3  # Should be a 3-channel color image
    assert img.dtype == np.uint8


def test_keyboard_outline_consistent():
    """Test that keyboard_outline is consistent with other constants."""
    # The keyboard width should match the expected total width
    expected_width = len(keyboard_geometry.white_keys) * keyboard_geometry.WHITE_BOTTOM_WIDTH
    assert keyboard_geometry.keyboard_outline["top-right"][0] == keyboard_geometry.KEYBOARD_WIDTH
    assert abs(expected_width - keyboard_geometry.KEYBOARD_WIDTH) < 0.1  # Allow for small floating point differences


def test_white_and_black_keys_complete():
    """Test that white_keys and black_keys together make up the complete keyboard."""
    all_keys = keyboard_geometry.white_keys + keyboard_geometry.black_keys
    all_keys.sort()
    expected_keys = list(range(21, 109))  # MIDI notes 21-108
    assert all_keys == expected_keys
