import cv2
import mido
import time
import os
import datetime
import threading
import numpy as np
from pathlib import Path
import json
import argparse

from analysis_hub import hub
import draw_keys_3d


# Define processing functions at the top of the file
def process_midi_event(event):
    """Process a single MIDI event and pass it to the analysis hub."""
    # Measure time spent in MIDI processing
    midi_start = time.time()
    hub.process_midi_event(event)
    midi_process_time = time.time() - midi_start
    if midi_process_time > 0.01:  # Log only if processing takes >10ms
        print(f"MIDI processing time: {midi_process_time*1000:.2f} ms")
    return hub.last_midi_result


def process_frame(frame):
    """Process a single video frame and pass it to the analysis hub."""
    # Time the frame processing
    start_time = time.time()
    hub.process_frame(frame)
    processing_time = time.time() - start_time

    # Get the processed output
    processed_frame = hub.last_image_output

    # Return both the processed frame and timing information
    return {
        "frame": processed_frame if processed_frame is not None else frame,
        "processing_time": processing_time,
    }


draw_keys_3d.init()

# Configuration
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FPS = 30


class MidiProcessor:
    def __init__(self, port_name=None):
        self.port_name = port_name
        self.processing = False
        self.start_time = None
        self.input_port = None
        self.midi_thread = None
        self.last_poll_time = 0
        self.poll_interval = 0.01

    def get_input_port(self):
        """Get the MIDI input port, either specified or first available."""
        available_ports = mido.get_input_names()
        if not available_ports:
            raise RuntimeError("No MIDI input ports available")

        if self.port_name is None:
            print("Available MIDI ports:")
            for i, port in enumerate(available_ports):
                print(f"{i}: {port}")

            selected_port = available_ports[0]
            print(f"Auto-selected: {selected_port}")
            return selected_port
        else:
            try:
                port_idx = int(self.port_name)
                if 0 <= port_idx < len(available_ports):
                    selected_port = available_ports[port_idx]
                    print(f"Selected port index {port_idx}: {selected_port}")
                    return selected_port
                else:
                    raise ValueError(f"MIDI port index {port_idx} out of range")
            except ValueError:
                # Not an integer, treat as port name
                if self.port_name not in available_ports:
                    raise ValueError(f"MIDI port '{self.port_name}' not found")
                return self.port_name

    def start_processing(self):
        """Start processing MIDI messages without recording."""
        port_name = self.get_input_port()
        self.input_port = mido.open_input(port_name)
        self.start_time = time.time()
        self.processing = True

        # Start the callback thread
        self.midi_thread = threading.Thread(target=self._process_callback)
        self.midi_thread.daemon = True
        self.midi_thread.start()

        print(f"Processing MIDI from {port_name}...")

    def _process_callback(self):
        """Thread function to process MIDI messages."""
        while self.processing:
            # Only poll at specific intervals to reduce CPU usage
            current_time = time.time()
            if current_time - self.last_poll_time < self.poll_interval:
                # Sleep to prevent busy-waiting
                time.sleep(0.0001)
                continue

            self.last_poll_time = current_time

            for msg in self.input_port.iter_pending():
                if not msg.is_meta:
                    assert self.start_time is not None
                    timestamp = time.time() - self.start_time
                    event = {"time": timestamp, "message": msg}
                    print(event)
                    # Use the process_midi_event function
                    process_midi_event(event)

    def stop_processing(self):
        """Stop processing and close the port."""
        self.processing = False
        if self.midi_thread:
            self.midi_thread.join(timeout=1.0)
            self.midi_thread = None
        if self.input_port:
            self.input_port.close()
            self.input_port = None
        print("MIDI processing stopped.")


class VideoProcessor:
    def __init__(self, device, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FPS):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.processing = False
        self.cap = None
        self.frame_count = 0
        self.display_window_name = "Video Processing"

    def start_processing(self):
        """Start processing video."""
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video device {self.device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")

        # Create a window for displaying the video
        cv2.namedWindow(self.display_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.display_window_name, 1280, 720
        )  # Resizable window with initial size

        self.processing = True
        self.frame_count = 0
        print("Video processing started")

    def process_frame(self):
        """Process a single video frame."""
        if not self.processing or not self.cap or not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        # Use the process_frame function defined at the top
        result = process_frame(frame)

        # Display the frame
        cv2.imshow(self.display_window_name, result["frame"])

        self.frame_count += 1
        print(f"Frame processing time: {result['processing_time']*1000:.2f} ms")

        return True

    def stop_processing(self):
        """Stop processing and release resources."""
        self.processing = False

        if self.cap:
            self.cap.release()
            self.cap = None

        # Close the display window
        cv2.destroyWindow(self.display_window_name)

        print(f"Video processing stopped after {self.frame_count} frames")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Live processing of MIDI and video.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--midi-port", type=str, default=None, help="MIDI port name or index to use"
    )
    args = parser.parse_args()

    midi_processor = MidiProcessor(port_name=args.midi_port)
    video_processor = VideoProcessor(device=args.camera)

    try:

        midi_processor.start_processing()
        video_processor.start_processing()

        print("Processing... Press 'q' to stop or Ctrl+C to interrupt.")

        # Process video frames in the main thread
        while video_processor.processing:
            if not video_processor.process_frame():
                break

            # Check for keyboard interrupt
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nProcessing stopped by user (q key).")
                break

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
    finally:
        # Stop processing
        midi_processor.stop_processing()
        video_processor.stop_processing()

        print("MIDI and video processing completed.")


if __name__ == "__main__":
    main()
