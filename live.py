"""
This module provides the graphical user interface (GUI) with the following functionality:

Camera and keyboard calibration.

Display of the live video stream, including visualization of incoming MIDI events.

Configuration of analysis modules to be executed.

Setup of OSC settings for further processing and sonification of analysis results.

Control over recording, allowing the user to start and stop capturing video, MIDI, and optionally audio.
"""

import tkinter as tk
from tkinter import ttk
import sys
import subprocess  # Added for running calibration scripts
import cv2  # Added for camera detection
import mido  # Added for MIDI device detection
import utils  # Added for camera orientation utilities
import threading
import time
import queue
from datetime import datetime


import analysis_hub
import draw_keys_3d

draw_keys_3d.init()


class PianoToolboxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Piano Toolbox")
        self.root.geometry("800x600")

        # Initialize variables
        self.available_cameras = self.get_available_cameras()
        self.selected_camera = tk.StringVar()
        if self.available_cameras:
            self.selected_camera.set("0")  # Default to first camera

        # Initialize MIDI variables
        self.available_midi_devices = self.get_available_midi_devices()
        self.selected_midi_device = tk.StringVar()
        if self.available_midi_devices:
            self.selected_midi_device.set(
                self.available_midi_devices[0]
            )  # Default to first device

        # MIDI monitoring variables
        self.midi_port = None
        self.midi_thread = None
        self.midi_monitoring = False
        self.midi_queue = queue.Queue()  # Remove maxsize limit to store all messages
        self.midi_lock = threading.Lock()  # Lock for thread safety

        # Initialize OSC variables
        self.osc_host = tk.StringVar(value="127.0.0.1")
        self.osc_port = tk.StringVar(value="9876")

        # Initialize stream control variables
        self.show_window = tk.BooleanVar(value=True)
        self.streaming = False
        self.cap = None
        self.stream_window_name = "Piano Toolbox Stream"
        self.stream_status = tk.StringVar(value="Inactive")

        # Add notification message variable
        self.notification_message = tk.StringVar()

        # For cleanup on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=1, fill="both", padx=10, pady=10)

        # Set up UI elements
        self.setup_ui()

        # Add notification label at the bottom of the window
        notification_label = ttk.Label(
            root, textvariable=self.notification_message, foreground="red"
        )
        notification_label.pack(side="bottom", fill="x", padx=10, pady=5)

        # Add style configuration for more compact widgets
        self.style = ttk.Style()
        self.style.configure("Compact.TButton", padding=2)
        self.style.configure("Compact.TCombobox", padding=2)

    def get_available_cameras(self):
        """Detect available camera devices using OpenCV"""
        available_cameras = []
        # Try to open each camera index from 0 to 10 (arbitrary upper limit)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()
            else:
                # Stop searching once we find a camera that doesn't open
                break
        return available_cameras

    def get_available_midi_devices(self):
        """Get list of available MIDI input devices using mido"""
        try:
            available_ports = mido.get_input_names()  # type: ignore
            return available_ports if available_ports else ["No MIDI devices detected"]
        except Exception as e:
            print(f"Error getting MIDI devices: {e}")
            return ["Error detecting MIDI devices"]

    def setup_ui(self):
        """Set up all UI elements in a single view"""
        # Calibration buttons section
        frame_calibration = ttk.LabelFrame(self.main_frame, text="Calibration")
        frame_calibration.pack(padx=0, pady=10, fill="x")

        # Create a frame for the calibration buttons to be side by side
        btn_frame = ttk.Frame(frame_calibration)
        btn_frame.pack(fill="x", padx=10, pady=5)

        # Button to configure camera orientation
        btn_camera_orientation = ttk.Button(
            btn_frame,
            text="Configure Camera Orientation",
            command=self.run_camera_orientation_calibration,
            style="Compact.TButton",
            width=25,
        )
        btn_camera_orientation.grid(row=0, column=0, padx=(0, 5), pady=5)

        # Button to configure keyboard geometry
        btn_keyboard_geometry = ttk.Button(
            btn_frame,
            text="Configure Keyboard Geometry",
            command=self.run_keyboard_calibration,
            style="Compact.TButton",
            width=25,
        )
        btn_keyboard_geometry.grid(row=0, column=1, padx=(5, 0), pady=5)

        # Make the columns expand equally
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # Camera settings
        frame_camera = ttk.LabelFrame(self.main_frame, text="Camera Settings")
        frame_camera.pack(padx=0, pady=10, fill="x")

        # Use grid for more control over layout
        grid_frame = ttk.Frame(frame_camera)
        grid_frame.pack(fill="x", padx=10, pady=5)

        # Camera index label and combobox
        ttk.Label(grid_frame, text="Camera index:").grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=2
        )

        # Camera selection combobox with indices
        camera_values = (
            self.available_cameras
            if self.available_cameras
            else ["No cameras detected"]
        )
        camera_combobox = ttk.Combobox(
            grid_frame,
            values=camera_values,
            textvariable=self.selected_camera,
            style="Compact.TCombobox",
            width=10,
        )
        camera_combobox.grid(row=0, column=1, sticky="w", pady=2)

        # Display a message about available cameras
        camera_info_text = (
            f"Found {len(self.available_cameras)} camera(s)"
            if self.available_cameras
            else "No cameras detected"
        )
        ttk.Label(grid_frame, text=camera_info_text).grid(
            row=0, column=2, sticky="w", padx=10, pady=2
        )

        # Configure the grid columns
        grid_frame.columnconfigure(2, weight=1)

        # MIDI settings
        frame_midi = ttk.LabelFrame(self.main_frame, text="MIDI Settings")
        frame_midi.pack(padx=0, pady=10, fill="x")

        # Use grid for MIDI settings
        midi_frame = ttk.Frame(frame_midi)
        midi_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(midi_frame, text="MIDI input device:").grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=2
        )

        # MIDI device selection combobox with available MIDI devices
        midi_combobox = ttk.Combobox(
            midi_frame,
            values=self.available_midi_devices,
            textvariable=self.selected_midi_device,
            style="Compact.TCombobox",
            width=25,
        )
        midi_combobox.grid(row=0, column=1, sticky="w", pady=2)

        # Display a message about available MIDI devices
        midi_info_text = (
            f"Found {len(self.available_midi_devices)} MIDI device(s)"
            if self.available_midi_devices
            and self.available_midi_devices[0] != "No MIDI devices detected"
            else "No MIDI devices detected"
        )
        ttk.Label(midi_frame, text=midi_info_text).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2
        )

        # Configure the grid columns
        midi_frame.columnconfigure(1, weight=1)

        # OSC settings
        frame_osc = ttk.LabelFrame(self.main_frame, text="OSC Settings")
        frame_osc.pack(padx=0, pady=10, fill="x")

        # Use grid for OSC settings
        osc_frame = ttk.Frame(frame_osc)
        osc_frame.pack(fill="x", padx=10, pady=5)

        # OSC Host
        ttk.Label(osc_frame, text="OSC Host:").grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=2
        )
        host_entry = ttk.Entry(osc_frame, width=15, textvariable=self.osc_host)
        host_entry.grid(row=0, column=1, sticky="w", pady=2)

        # OSC Port
        ttk.Label(osc_frame, text="OSC Port:").grid(
            row=1, column=0, sticky="w", padx=(0, 5), pady=2
        )
        port_entry = ttk.Entry(osc_frame, width=15, textvariable=self.osc_port)
        port_entry.grid(row=1, column=1, sticky="w", pady=2)

        # Configure the grid columns
        osc_frame.columnconfigure(1, weight=1)

        # Stream control section
        frame_controls = ttk.LabelFrame(self.main_frame, text="Stream Control")
        frame_controls.pack(padx=0, pady=10, fill="x")

        control_frame = ttk.Frame(frame_controls)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Stream status indicator
        status_label = ttk.Label(control_frame, text="Stream Status:")
        status_label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)

        # Status value with color formatting
        status_value = ttk.Label(
            control_frame, textvariable=self.stream_status, foreground="red"
        )
        status_value.grid(row=0, column=1, sticky="w", pady=2)

        # Store the label to update its color
        self.status_label = status_value

        # Show window checkbox
        show_window_check = ttk.Checkbutton(
            control_frame, text="Show Window", variable=self.show_window
        )
        show_window_check.grid(row=0, column=2, sticky="e", padx=(20, 5), pady=2)

        # Start Stream button
        btn_start = ttk.Button(
            control_frame, text="Start Stream", command=self.run_live_runner
        )
        btn_start.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)

        # Configure the grid columns
        control_frame.columnconfigure(2, weight=1)

    def run_camera_orientation_calibration(self):
        """Run the external camera orientation calibration script"""
        try:
            subprocess.Popen(["python", "calibrate_camera_orientation.py"])
        except Exception as e:
            print(f"Error running camera calibration: {e}")

    def run_keyboard_calibration(self):
        """Run the external keyboard geometry calibration script"""
        try:
            subprocess.Popen(["python", "calibrate_keyboard.py"])
        except Exception as e:
            print(f"Error running keyboard calibration: {e}")

    def run_live_runner(self):
        """Run the live_runner.py script with selected parameters"""
        if self.streaming:
            return  # Already streaming

        # Clear any existing notification
        self.notification_message.set("")

        # Update status
        self.streaming = True
        self.stream_status.set("Active")
        self.status_label.config(foreground="green")

        try:
            camera_index = int(self.selected_camera.get())
            midi_device = self.selected_midi_device.get()

            # Prepare command to run live_runner.py
            cmd = ["python", "live_runner.py", "--camera", str(camera_index)]

            # Add MIDI port if valid
            if midi_device and midi_device not in [
                "No MIDI devices detected",
                "Error detecting MIDI devices",
            ]:
                cmd.extend(["--midi-port", midi_device])

            # Start live_runner.py as a subprocess
            print(f"Running command: {' '.join(cmd)}")
            self.live_process = subprocess.Popen(cmd)

            # Check process status periodically
            self.root.after(1000, self.check_live_process)

        except Exception as e:
            print(f"Error starting live_runner: {e}")
            self.notification_message.set(f"Error starting live_runner: {e}")
            self.stop_live_runner()

    def check_live_process(self):
        """Check if the live_runner process is still running"""
        if hasattr(self, "live_process"):
            assert self.live_process is not None
            retcode = self.live_process.poll()
            if retcode is not None:
                # Process has ended
                if retcode != 0:
                    self.notification_message.set(
                        f"live_runner exited with code {retcode}"
                    )
                self.stop_live_runner()
            else:
                # Process still running, check again later
                self.root.after(1000, self.check_live_process)

    def stop_live_runner(self):
        """Stop the live_runner process"""
        if hasattr(self, "live_process") and self.live_process:
            try:
                self.live_process.terminate()
                self.live_process = None
            except Exception as e:
                print(f"Error terminating live_runner: {e}")

        # Update UI
        self.streaming = False
        self.stream_status.set("Inactive")
        self.status_label.config(foreground="red")

    def start_midi_monitoring(self):
        """Start monitoring MIDI input in a separate thread"""
        if self.midi_monitoring:
            return  # Already monitoring

        device_name = self.selected_midi_device.get()
        if not device_name or device_name in [
            "No MIDI devices detected",
            "Error detecting MIDI devices",
        ]:
            self.notification_message.set("No valid MIDI device selected")
            return

        try:
            # Stop any existing monitoring
            self.stop_midi_monitoring()

            # Start MIDI monitoring in a separate thread
            self.midi_monitoring = True
            self.midi_thread = threading.Thread(
                target=self.midi_monitor_thread, daemon=True
            )
            self.midi_thread.start()

            print(f"Started MIDI monitoring on device: {device_name}")
        except Exception as e:
            self.notification_message.set(f"Error starting MIDI monitoring: {e}")
            print(f"Error starting MIDI monitoring: {e}")

    def midi_monitor_thread(self):
        """Thread function to monitor MIDI input"""
        device_name = self.selected_midi_device.get()

        try:
            # Open the MIDI input port
            self.midi_port = mido.open_input(device_name)  # type: ignore

            # Monitor for MIDI messages
            while self.midi_monitoring:
                try:
                    # Use callback approach instead of polling to avoid GIL issues
                    msg = self.midi_port.receive(block=False)
                    if msg is not None:
                        # Format and add timestamp
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        formatted_msg = f"{timestamp}: {msg}"
                        print(formatted_msg)  # For debugging
                        e = {
                            "timestamp": 0.0,
                            "message": msg,
                        }
                        analysis_hub.hub.process_midi_event(
                            e
                        )  # Process the MIDI message
                except Exception as e:
                    # Only print error if still monitoring
                    if self.midi_monitoring:
                        print(f"Error receiving MIDI message: {e}")

                # Sleep to prevent high CPU usage
                time.sleep(0.01)

        except Exception as e:
            print(f"Error in MIDI monitoring thread: {e}")
        finally:
            # Clean up
            if hasattr(self, "midi_port") and self.midi_port:
                try:
                    self.midi_port.close()
                except Exception as e:
                    print(f"Error closing MIDI port: {e}")
                self.midi_port = None

    def stop_midi_monitoring(self):
        """Stop the MIDI monitoring thread"""
        # Set flag first
        self.midi_monitoring = False

        # Give thread a moment to notice the flag change
        time.sleep(0.1)

        # Close the port
        if self.midi_port:
            try:
                self.midi_port.close()
            except Exception as e:
                print(f"Error closing MIDI port: {e}")
            self.midi_port = None

        # Wait for thread to finish
        if self.midi_thread and self.midi_thread.is_alive():
            try:
                self.midi_thread.join(timeout=1.0)
                print("MIDI monitoring stopped")
            except Exception as e:
                print(f"Error joining MIDI thread: {e}")

    def start_stream(self):
        """Start the camera stream with the selected camera"""
        if self.streaming:
            return  # Already streaming

        # Clear any existing notification
        self.notification_message.set("")

        # Start MIDI monitoring first
        self.start_midi_monitoring()

        try:
            camera_index = int(self.selected_camera.get())
            self.cap = cv2.VideoCapture(camera_index)

            if not self.cap.isOpened():
                print(f"Error: Could not open camera {camera_index}")
                self.notification_message.set(
                    f"Error: Could not open camera {camera_index}"
                )
                # Stop MIDI if camera fails
                self.stop_midi_monitoring()
                return

            self.streaming = True
            self.stream_status.set("Active")
            self.status_label.config(foreground="green")
            self.update_frame()
            print(f"Started streaming from camera {camera_index}")

        except Exception as e:
            print(f"Error starting stream: {e}")
            self.notification_message.set(f"Error starting stream: {e}")
            # Stop MIDI if camera fails
            self.stop_midi_monitoring()

    def stop_stream(self):
        """Stop the camera stream"""
        self.streaming = False
        self.stream_status.set("Inactive")
        self.status_label.config(foreground="red")

        # Stop MIDI monitoring when stream stops
        self.stop_midi_monitoring()

        if self.cap:
            self.cap.release()
            self.cap = None

        # Safely close any open windows
        try:
            cv2.destroyWindow(self.stream_window_name)
        except:
            pass  # Window might not exist, that's fine

        print("Stopped streaming")

    def update_frame(self):
        """Update the video frame and process it"""
        if not self.streaming or not self.cap:
            return

        ret, frame = self.cap.read()

        if not ret:
            print("Failed to grab frame from camera. Stopping stream.")
            self.stop_stream()
            return

        # Process the frame here - you can add more processing as needed
        frame = utils.flip_image(frame)  # Apply camera orientation settings
        analysis_hub.hub.process_frame(frame)  # Process the frame through analysis hub

        # Safely handle window visibility
        try:
            if self.show_window.get():
                # Check if hub.last_image_output is None and use frame instead if it is
                display_frame = (
                    analysis_hub.hub.last_image_output
                    if analysis_hub.hub.last_image_output is not None
                    else frame
                )
                cv2.imshow(self.stream_window_name, display_frame)
                cv2.waitKey(1)  # Required for the window to update
            else:
                # Only try to close if we know the window was previously open
                try:
                    cv2.destroyWindow(self.stream_window_name)
                except:
                    pass  # Window might not exist, that's fine
        except Exception as e:
            print(f"Error updating window: {e}")

        # Schedule the next frame update
        self.root.after(15, self.update_frame)  # ~60 FPS

    def on_closing(self):
        """Handle window close event"""
        # Stop the live_runner process if running
        if hasattr(self, "live_process") and self.live_process:
            self.stop_live_runner()

        # First stop streaming which includes stopping MIDI
        self.stop_stream()

        # Cancel any pending after() calls
        for after_id in self.root.tk.call("after", "info"):
            self.root.after_cancel(after_id)

        # Now destroy the window
        self.root.destroy()

        # Force exit to prevent hanging threads
        import os

        os._exit(0)


def main():
    root = tk.Tk()
    app = PianoToolboxApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
