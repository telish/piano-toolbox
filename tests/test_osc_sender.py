import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import osc_sender


def test_configure():
    """Test that configure sets up the client correctly."""
    with patch("pythonosc.udp_client.SimpleUDPClient") as mock_client:
        # Store the original SimpleUDPClient class
        original_client = osc_sender.SimpleUDPClient

        try:
            # Replace the SimpleUDPClient class with our mock
            osc_sender.SimpleUDPClient = mock_client

            # Call configure with a test port
            test_port = 12345
            osc_sender.configure(test_port)

            # Check that SimpleUDPClient was created with correct parameters
            mock_client.assert_called_once_with("127.0.0.1", test_port)

            # Check that global variables were set
            assert osc_sender._state["osc_port"] == test_port
            assert osc_sender._state["osc_client"] is not None
        finally:
            # Restore the original SimpleUDPClient class
            osc_sender.SimpleUDPClient = original_client


def test_send_message():
    """Test that send_message passes the correct data to the client."""
    # Create a mock client
    mock_client = MagicMock()

    # Replace the global client with our mock
    with patch.object(osc_sender, "_state", {"osc_client": mock_client}):
        # Test sending a message with no args
        osc_sender.send_message("/test/address")
        mock_client.send_message.assert_called_with("/test/address", ())

        # Test sending a message with single arg
        osc_sender.send_message("/test/address", 42)
        mock_client.send_message.assert_called_with("/test/address", (42,))

        # Test sending a message with multiple args
        osc_sender.send_message("/test/address", 1, 2, 3, "test")
        mock_client.send_message.assert_called_with("/test/address", (1, 2, 3, "test"))
