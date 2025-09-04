"""Checks video rate (fps) of a recording by analyzing the timestamps.json file."""

import json
import matplotlib.pyplot as plt


def parse_timestamps(json_path: str) -> list[dict]:
    """
    Liest die Timestamps aus der JSON-Datei.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_inter_frame_intervals(data: list[dict], bins: int = 50):
    """
    Berechnet die Zeitintervalle zwischen Frames und zeigt ein Histogramm.

    Args:
        data (list of dict): Liste von Dicts mit "timestamp" Schlüssel
        bins (int): Anzahl der Bins im Histogramm (default: 50)
    """
    timestamps = [d["timestamp"] for d in data]
    timestamps.sort()

    frame_rate = [1.0 / (t2 - t1) for t1, t2 in zip(timestamps[:-1], timestamps[1:])]

    plt.figure(figsize=(10, 6))
    plt.hist(frame_rate, bins=bins, edgecolor="black")
    plt.xlabel("Frame Rate (fps)")
    plt.ylabel("Häufigkeit")
    plt.title(f"Verteilung der Frame-Raten\n{len(frame_rate)} Frames")
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    timestamps_path = "recording/video/timestamps.json"
    timestamps = parse_timestamps(timestamps_path)
    plot_inter_frame_intervals(timestamps)


if __name__ == "__main__":
    main()
