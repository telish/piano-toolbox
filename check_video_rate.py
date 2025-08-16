
import os
import matplotlib.pyplot as plt


def parse_video(path):
    video_frames = []
    for filename in os.listdir(path):
        if filename.startswith("frame_") and filename.endswith(".png"):
            num_as_string = filename[6:-4].replace('_', '.')
            try:
                timestamp = float(num_as_string)
                video_frames.append(
                    {'timestamp': timestamp, 'type': 'video', 'filename': filename})
            except ValueError as e:
                print(f"Error parsing filename: {filename} -> {e}")
    return video_frames


def plot_inter_frame_intervals(data, bins=50):
    """
    Calculates the inter-frame intervals (differences between adjacent timestamps)
    and plots a histogram.

    Args:
        data (list of dict): List of dicts, each containing a "timestamp" key.
        bins (int): Number of bins in the histogram (default: 50).
    """
    timestamps = [d["timestamp"] for d in data]
    timestamps.sort()

    frame_rate = [1./(t2 - t1) for t1, t2 in zip(timestamps[:-1], timestamps[1:])]

    plt.hist(frame_rate, bins=bins, edgecolor='black')
    plt.xlabel("Frame Rate (fps)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Frame Rates")
    plt.show()


video_path = "recording/video"
video_events = parse_video(video_path)
plot_inter_frame_intervals(video_events)
