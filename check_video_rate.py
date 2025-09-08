"""Checks the video frame rate (fps) of a recording by analyzing the timestamps.json file."""

import json

import matplotlib.pyplot as plt


def plot_inter_frame_intervals(data: list[dict], bins: int = 50) -> None:
    """
    Calculates the time intervals between frames and displays a histogram.

    Args:
        data (list of dict): List of dicts with a "timestamp" key
        bins (int): Number of bins in the histogram (default: 50)
    """
    timestamps = [d["timestamp"] for d in data]
    timestamps.sort()

    frame_rate = [1.0 / (t2 - t1) for t1, t2 in zip(timestamps[:-1], timestamps[1:])]

    plt.figure(figsize=(10, 6))
    plt.hist(frame_rate, bins=bins, edgecolor="black")
    plt.xlabel("Frame Rate (fps)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Frame Rates\n{len(frame_rate)} Frames")
    plt.grid(True, alpha=0.3)
    plt.show()


def main() -> None:
    json_path = "recording/video/timestamps.json"
    with open(json_path, "r", encoding="utf-8") as f:
        timestamps = json.load(f)

    plot_inter_frame_intervals(timestamps)


if __name__ == "__main__":
    main()
