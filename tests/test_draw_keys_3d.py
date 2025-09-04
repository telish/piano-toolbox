import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import draw_keys_3d
import keyboard_geometry
import utils


@pytest.fixture
def mock_homography_matrix():
    """Create a mock homography matrix for testing."""
    # Simple identity matrix with some scaling
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


@pytest.fixture
def mock_keypoint_mappings():
    """Create mock keypoint mappings for testing."""
    return [
        {"pixel": (100, 100), "object": (0.0, 0.0)},
        {"pixel": (200, 100), "object": (keyboard_geometry.WHITE_BOTTOM_WIDTH, 0.0)},
        {
            "pixel": (200, 200),
            "object": (
                keyboard_geometry.WHITE_BOTTOM_WIDTH,
                keyboard_geometry.WHITE_HEIGHT,
            ),
        },
        {"pixel": (100, 200), "object": (0.0, keyboard_geometry.WHITE_HEIGHT)},
    ]


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
            {"pixel": [300, 100], "object": [100.0, 0.0]},
            {"pixel": [300, 300], "object": [100.0, 100.0]},
            {"pixel": [100, 300], "object": [0.0, 100.0]},
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


def test_init_with_provided_keypoints(mock_keypoint_mappings, mock_homography_matrix):
    """Test initializing with provided keypoint mappings."""
    with patch("cv2.findHomography", return_value=(mock_homography_matrix, None)):
        draw_keys_3d.init(mock_keypoint_mappings)
        assert draw_keys_3d.homography_matrix is not None
        assert np.array_equal(draw_keys_3d.homography_matrix, mock_homography_matrix)


def test_init_loads_from_file(mock_keyboard_geometry_file, mock_homography_matrix):
    """Test that init loads keypoint mappings from file when none are provided."""
    with patch(
        "utils.get_keyboard_geometry_file_path",
        return_value=os.path.join(
            mock_keyboard_geometry_file,
            "calibration",
            "keyboard",
            "keyboard_geometry.json",
        ),
    ), patch("cv2.findHomography", return_value=(mock_homography_matrix, None)):
        draw_keys_3d.init()
        assert draw_keys_3d.homography_matrix is not None


def test_draw_polygon():
    """Test drawing a polygon on an image."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    points = np.array(
        [[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]], dtype=np.float32
    )
    color = (0, 255, 0)

    with patch("cv2.polylines") as mock_polylines:
        draw_keys_3d.draw_polygon(img, points, color)
        mock_polylines.assert_called_once()


def test_pixel_coordinates_of_key(mock_homography_matrix):
    """Test computing pixel coordinates of a key."""
    draw_keys_3d.homography_matrix = mock_homography_matrix

    with patch(
        "keyboard_geometry.key_points",
        return_value=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    ):
        coords = draw_keys_3d.pixel_coordinates_of_key(60)  # Middle C
        assert coords.shape[0] == 4  # 4 points for the key
        assert coords.shape[1] == 1  # Each point is 1x2
        assert coords.shape[2] == 2  # x,y coordinates


def test_pixel_coordinates_of_bounding_box(mock_homography_matrix):
    """Test computing pixel coordinates of a key's bounding box."""
    draw_keys_3d.homography_matrix = mock_homography_matrix

    with patch(
        "keyboard_geometry.key_bounding_box",
        return_value=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    ):
        coords = draw_keys_3d.pixel_coordinates_of_bounding_box(60)  # Middle C
        assert coords.shape[0] == 4  # 4 points for the bounding box
        assert coords.shape[1] == 1  # Each point is 1x2
        assert coords.shape[2] == 2  # x,y coordinates


def test_draw_key(mock_homography_matrix):
    """Test drawing a key on an image."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    draw_keys_3d.homography_matrix = mock_homography_matrix
    midi_pitch = 60  # Middle C
    color = (0, 255, 0)

    with patch(
        "draw_keys_3d.pixel_coordinates_of_key",
        return_value=np.array(
            [[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]], dtype=np.float32
        ),
    ), patch("draw_keys_3d.draw_polygon"), patch("draw_keys_3d.draw_annotation"):
        result = draw_keys_3d.draw_key(img, midi_pitch, color, "C4")
        assert result is img  # Should return the same image


def test_draw_keyboard(mock_homography_matrix):
    """Test drawing the entire keyboard."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    draw_keys_3d.homography_matrix = mock_homography_matrix
    color = (0, 255, 0)

    with patch("draw_keys_3d.draw_key") as mock_draw_key:
        result = draw_keys_3d.draw_keyboard(img, color)
        assert result is img  # Should return the same image
        assert mock_draw_key.call_count == 88  # Should draw all 88 keys


def test_draw_annotation():
    """Test drawing an annotation on a key."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    midi_pitch = 60  # Middle C
    color = (0, 255, 0)
    annotation = "C4"
    image_points = np.array(
        [[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]], dtype=np.float32
    )

    with patch("cv2.getTextSize", return_value=((30, 20), 5)), patch(
        "cv2.rectangle"
    ), patch("cv2.putText"):
        draw_keys_3d.draw_annotation(img, midi_pitch, color, annotation, image_points)
        # Since we're mocking all cv2 calls, just verify the function completes


def test_draw_annotation_black_key():
    """Test drawing an annotation on a black key, which has different y_offset."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    midi_pitch = 61  # C# (black key)
    color = (0, 255, 0)
    annotation = "C#4"
    image_points = np.array(
        [[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]], dtype=np.float32
    )

    with patch("keyboard_geometry.BLACK_KEYS", [61]), patch(
        "cv2.getTextSize", return_value=((30, 20), 5)
    ), patch("cv2.rectangle"), patch("cv2.putText"):
        draw_keys_3d.draw_annotation(img, midi_pitch, color, annotation, image_points)
        # Since we're mocking all cv2 calls, just verify the function completes


@patch("cv2.imshow")
@patch("cv2.waitKey")
@patch("cv2.destroyAllWindows")
def test_main(
    mock_destroy,
    mock_waitkey,
    mock_imshow,
    mock_keyboard_geometry_file,
    mock_homography_matrix,
):
    """Test the main function runs without errors."""
    # Set up mocks for init and image loading
    with patch("draw_keys_3d.init"), patch(
        "utils.get_keyboard_image_file_path",
        return_value=os.path.join(
            mock_keyboard_geometry_file, "calibration", "keyboard", "foto.png"
        ),
    ), patch("cv2.imread", return_value=np.zeros((300, 300, 3), dtype=np.uint8)), patch(
        "utils.flip_image", return_value=np.zeros((300, 300, 3), dtype=np.uint8)
    ), patch(
        "draw_keys_3d.draw_key"
    ):

        # Mock waitKey to return immediately
        mock_waitkey.return_value = 0

        # Run the main function
        draw_keys_3d.main()

        # Verify that the expected functions were called
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once()
        mock_destroy.assert_called_once()


def test_main_image_not_found():
    """Test that main raises FileNotFoundError when image is not found."""
    with patch("draw_keys_3d.init"), patch(
        "utils.get_keyboard_image_file_path", return_value="nonexistent_path.png"
    ), patch("cv2.imread", return_value=None):

        with pytest.raises(FileNotFoundError):
            draw_keys_3d.main()


def test_homography_not_initialized():
    """Test that functions requiring homography matrix raise assertion error when not initialized."""
    # Reset homography matrix
    draw_keys_3d.homography_matrix = None

    with pytest.raises(AssertionError):
        draw_keys_3d.pixel_coordinates_of_key(60)

    with pytest.raises(AssertionError):
        draw_keys_3d.pixel_coordinates_of_bounding_box(60)


def test_init_invalid_keypoints():
    """Test init with invalid keypoint mappings."""
    # Empty keypoint mappings
    with pytest.raises(AssertionError):
        draw_keys_3d.init([])
