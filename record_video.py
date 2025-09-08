"""Record video from camera."""

import argparse
import json
import os
import signal
import sys
import time

import cv2


def parse_args() -> argparse.Namespace:
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
        default="recording",
        help="Directory to save video (default: recording)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Start recording video.

    Args:
        camera_index: Optional index of the camera to use
        show_image: Optional flag to show image during recording
        output_dir: Optional output directory for the video files
    """
    args = parse_args()

    # Create output directory
    video_dir = os.path.join(args.output_dir, "video")
    os.makedirs(video_dir, exist_ok=True)

    # Configure video capture
    fourcc = cv2.VideoWriter_fourcc(*"HFYU")  # HuffYUV Codec (lossless)  # type: ignore
    FPS = 30.0
    video_filename = os.path.join(video_dir, "recording.avi")
    timestamps_filename = os.path.join(video_dir, "timestamps.json")

    # Open camera
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("record-video.py: Error. Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Init video writer
    ret, frame = cap.read()
    if not ret:
        print("record-video.py: Error. Failed to capture first frame.")
        sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FPS, FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")

    height, width = frame.shape[:2]
    out = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))
    timestamps = []

    # Signal handler for graceful exit
    def signal_handler(_sig: int, _frame: object) -> None:
        out.release()
        cap.release()
        cv2.destroyAllWindows()

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
            timestamps.append({"timestamp": time.time(), "frame_number": len(timestamps)})

            if args.show_image:
                cv2.imshow("Recording", frame)

            # Exit recording when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                signal_handler(signal.SIGINT, None)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
