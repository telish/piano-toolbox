"""Simulate recording by playing back recorded video and MIDI messages."""

import json
import os

# import time
import argparse  # For command-line arguments
import time

import cv2
import mido

from analysis_hub import hub
import draw_keys_3d
import utils
import osc_sender


def parse_midi_msgs(filename):
    """Parse MIDI messages from file. Returns empty list if file doesn't exist."""
    result = []
    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    timestamp, msg_text = line.strip().split(": ", 1)
                    timestamp = float(timestamp)  # Convert to float
                    # Convert text to mido Message
                    msg = mido.Message.from_str(msg_text)
                    result.append(
                        {"timestamp": timestamp, "type": "midi", "message": msg}
                    )
                except ValueError as e:
                    print(f"Error parsing line: {line} -> {e}")
    except FileNotFoundError:
        print(f"MIDI file not found: {filename}")
        return []

    return result


def parse_video_timestamps(timestamps_file):
    """Parse the timestamps.json file created by record_video.py"""
    with open(timestamps_file, "r", encoding="utf-8") as f:
        timestamps = json.load(f)
    return [
        {
            "timestamp": t["timestamp"],
            "type": "video",
            "frame_number": t["frame_number"],
        }
        for t in timestamps
    ]


def parse_video(path):
    """Read video file and timestamps"""
    # Read timestamps
    timestamps_file = os.path.join(path, "timestamps.json")
    if not os.path.exists(timestamps_file):
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")

    # Open video file
    video_file = os.path.join(path, "recording.avi")
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")

    return parse_video_timestamps(timestamps_file)


def draw_text(img, instruction_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(instruction_text, font, 0.7, 2)[0]

    # Position at bottom of the image
    text_x = 10
    text_y = img.shape[0] - 20

    # Draw background rectangle
    cv2.rectangle(
        img,
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        (0, 0, 0),
        -1,
    )

    # Draw text
    cv2.putText(
        img,
        instruction_text,
        (text_x, text_y),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Update the display with the text
    cv2.imshow("Simulate Recording", img)


skip_to_next_note = {
    "should_skip_to_next_note": False,
    "note_received": False,
    "should_skip_to_end": False,
}


def handle_keyboard_input(img):
    global skip_to_next_note
    stop = not skip_to_next_note["should_skip_to_end"] and (
        (not skip_to_next_note["should_skip_to_next_note"])
        or (
            skip_to_next_note["should_skip_to_next_note"]
            and skip_to_next_note["note_received"]
        )
    )
    if stop:
        # Draw instructions directly on the image
        instruction_text = "Press 'f' for next frame, 'n' for the next note, 'e' to continue until the end, 'q' to quit"
        draw_text(img, instruction_text)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("f"):
                break
            if key == ord("n"):
                skip_to_next_note = {
                    "should_skip_to_next_note": True,
                    "note_received": False,
                    "should_skip_to_end": False,
                }
                break
            elif key == ord("e"):
                skip_to_next_note["should_skip_to_end"] = True
                break
            elif key == ord("q"):
                print("Quitting simulation.")
                cv2.destroyAllWindows()
                exit(0)
    else:
        cv2.waitKey(1)  # Just to update the window without blocking


def get_all_events(recording_base):
    midi_events = parse_midi_msgs(os.path.join(recording_base, "midi/midi_msg.txt"))
    video_events = parse_video(os.path.join(recording_base, "video"))

    # Combine and sort events by timestamp
    all_events = midi_events + video_events
    all_events.sort(key=lambda event: event["timestamp"])

    return all_events


def process_midi(event):
    print(f"{event['timestamp']}: {event['message']}")
    hub.process_midi_event(event)
    res = hub.last_midi_result
    if hub.last_midi_result["msg.type"] == "note_on":
        print(res["hand"].capitalize(), res["finger"])
        if skip_to_next_note["should_skip_to_next_note"]:
            skip_to_next_note["note_received"] = True


class VideoPlayer:
    """Class to handle video playback."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.video_capture = None

    def load_video_if_needed(self):
        if self.video_capture is None:
            video_file = os.path.join(self.video_path, "video", "recording.avi")
            self.video_capture = cv2.VideoCapture(video_file)

    def read_frame(self, frame_number):
        self.load_video_if_needed()
        assert self.video_capture is not None, "Video capture is not initialized"
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = self.video_capture.read()
        return ret, img


def process_video_frame(event, video_processor):
    print(f"{event['timestamp']:.7f}: Frame {event['frame_number']}")

    # Read frame
    ret, img = video_processor.read_frame(event["frame_number"])
    if not ret:
        print(f"Failed to read frame {event['frame_number']}")
        return

    # Process frame
    img = utils.flip_image(img)

    # Time the frame processing
    # start_time = time.time()
    hub.process_frame(img)
    # processing_time = time.time() - start_time
    # print(f"Frame processing time: {processing_time*1000:.2f} ms")

    img = hub.last_image_output
    assert img is not None, "No image output from hub"

    hub.draw_results(img)

    cv2.imshow("Simulate Recording", img)
    handle_keyboard_input(img)


def main():
    parser = argparse.ArgumentParser(description="Simulate recording playback.")
    parser.add_argument(
        "--port-out",
        default=9876,
        type=int,
        help="OSC outgoing port (default: 9876)",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default="./recording",
        help="Path of the recording",
    )
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.recording, "calibration")):
        utils.set_calibration_base_dir(args.recording)
    osc_sender.configure(args.port_out)
    draw_keys_3d.init()

    video_player = VideoPlayer(args.recording)
    all_events = get_all_events(args.recording)
    # start_real = time.time()
    # start_recording = all_events[0]["timestamp"]
    for event in all_events:
        # Wait until the event's timestamp is reached
        # time_to_sleep = event["timestamp"] - start_recording - (time.time() - start_real)
        # if time_to_sleep < 0:  # Make sure time to sleep is not negative.
        #     time_to_sleep = 0
        # time.sleep(time_to_sleep)

        if event["type"] == "midi":
            process_midi(event)
        elif event["type"] == "video":
            process_video_frame(event, video_player)
        else:
            assert False, f"Unknown event type: {event['type']}"

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
