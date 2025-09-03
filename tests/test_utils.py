import os
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    set_calibration_base_dir,
    get_keyboard_image_file_path,
    get_keyboard_geometry_file_path,
    flip_image,
    add_text_to_image,
)


@pytest.fixture
def mock_base_dir(tmp_path):
    """Create a temporary directory structure for testing."""
    # Create the calibration directory structure
    calibration_dir = tmp_path / "calibration"
    keyboard_dir = calibration_dir / "keyboard"
    keyboard_dir.mkdir(parents=True)

    # Create mock files
    mock_keyboard_image = keyboard_dir / "foto.png"
    mock_keyboard_image.write_bytes(b"mock image data")

    mock_keyboard_geometry = keyboard_dir / "keyboard_geometry.json"
    mock_keyboard_geometry.write_text('{"mock": "data"}')

    mock_camera_orientation = calibration_dir / "camera_orientation.json"
    mock_camera_orientation.write_text('{"flip_horizontal": true, "flip_vertical": false}')

    return tmp_path


def test_set_calibration_base_dir(mock_base_dir):
    """Test setting the calibration base directory."""
    set_calibration_base_dir(str(mock_base_dir))

    # Verify the paths are correct
    assert get_keyboard_image_file_path() == os.path.join(str(mock_base_dir), "calibration", "keyboard", "foto.png")
    assert get_keyboard_geometry_file_path() == os.path.join(
        str(mock_base_dir), "calibration", "keyboard", "keyboard_geometry.json"
    )


def test_flip_image():
    """Test the image flipping functionality."""
    # Create a simple test image
    test_image = np.zeros((3, 3), dtype=np.uint8)
    test_image[0, 0] = 255  # Set top-left pixel to white

    # Test with horizontal flip
    with patch("utils.flip_horizontal", True), patch("utils.flip_vertical", False):
        flipped = flip_image(test_image)
        assert flipped[0, 2] == 255  # Top-right should be white after horizontal flip
        assert flipped[0, 0] == 0  # Top-left should be black

    # Test with vertical flip
    with patch("utils.flip_horizontal", False), patch("utils.flip_vertical", True):
        flipped = flip_image(test_image)
        assert flipped[2, 0] == 255  # Bottom-left should be white after vertical flip
        assert flipped[0, 0] == 0  # Top-left should be black

    # Test with both flips
    with patch("utils.flip_horizontal", True), patch("utils.flip_vertical", True):
        flipped = flip_image(test_image)
        assert flipped[2, 2] == 255  # Bottom-right should be white after both flips
        assert flipped[0, 0] == 0  # Top-left should be black


def test_add_text_to_image():
    """Test adding text to an image."""
    # Create a test image
    img = np.zeros((200, 400, 3), dtype=np.uint8)

    # Test adding single line text
    result = add_text_to_image(img.copy(), "Test text", position="bottom-left")
    # Check if the image was modified (simple check - not completely black anymore)
    assert np.sum(result) > 0

    # Test adding multiline text
    result = add_text_to_image(img.copy(), "Line 1\nLine 2", position="top-right")
    assert np.sum(result) > 0

    # Test with custom max width to force line wrapping
    long_text = "This is a very long text that should be wrapped automatically"
    result = add_text_to_image(img.copy(), long_text, max_text_width=100)
    assert np.sum(result) > 0

    # Test different positions
    positions = ["bottom-left", "bottom-right", "top-left", "top-right"]
    for pos in positions:
        result = add_text_to_image(img.copy(), "Position test", position=pos)
        assert np.sum(result) > 0
