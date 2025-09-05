"""Record video from camera."""

import argparse
import json
import os
import signal
import sys
import time

import cv2


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Record video from webcam.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the camera to use (default: 0)",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show the video frames during recording",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recording/video",
        help="Directory to save video (default: recording/video)",
    )
    return parser.parse_args()


def main(camera_index=None, show_image=None, output_dir=None):
    """
    Start recording video.

    Args:
        camera_index: Optional index of the camera to use
        show_image: Optional flag to show image during recording
        output_dir: Optional output directory for the video files
    """
    # Use provided arguments or parse from command line
    if camera_index is None or show_image is None or output_dir is None:
        args = parse_args()
        camera_index = args.camera_index if camera_index is None else camera_index
        show_image = args.show_image if show_image is None else show_image
        output_dir = args.output_dir if output_dir is None else output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure video capture
    fourcc = cv2.VideoWriter_fourcc(*"HFYU")  # HuffYUV Codec (lossless)  # type: ignore
    FPS = 30.0
    video_filename = os.path.join(output_dir, "recording.avi")
    timestamps_filename = os.path.join(output_dir, "timestamps.json")

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("record-video.py: Error. Could not open webcam.")
        sys.exit(1)

    # Init video writer
    ret, frame = cap.read()
    if not ret:
        print("record-video.py: Error. Failed to capture first frame.")
        sys.exit(1)

    height, width = frame.shape[:2]
    out = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))
    timestamps = []

    # Signal handler for graceful exit
    def signal_handler(_sig, _frame):
        # Close video writer and camera
        out.release()
        cap.release()
        cv2.destroyAllWindows()

        # Save timestamps to file
        with open(timestamps_filename, "w", encoding="utf-8") as f:
            json.dump(timestamps, f)

        print(f"record-video.py: Video saved to {video_filename}")
        print(f"record-video.py: Timestamps saved to {timestamps_filename}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("record-video.py: Recording camera. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("record-video.py: Error. Failed to capture frame.")
                break

            # Save frame and timestamp
            out.write(frame)
            timestamps.append(
                {"timestamp": time.time(), "frame_number": len(timestamps)}
            )

            # Show the image in a window if requested
            if show_image:
                cv2.imshow("Recording", frame)

            # Exit recording when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                signal_handler(signal.SIGINT, None)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

    # Cleanup (in case loop exits normally)
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
