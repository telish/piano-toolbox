import subprocess

import cv2
import mido
import ttkbootstrap as ttk
from ttkbootstrap.constants import INFO, PRIMARY, SUCCESS

import draw_keys_3d

draw_keys_3d.init()


class PianoToolboxApp:
    def __init__(self, root: ttk.Window) -> None:
        self.root = root
        self.root.title("Piano Toolbox")
        self.root.geometry("800x600")

        # Variables
        self.available_cameras = self.get_available_cameras()
        self.selected_camera = ttk.StringVar()
        if self.available_cameras:
            self.selected_camera.set(self.available_cameras[0])

        self.available_midi_devices = self.get_available_midi_devices()
        self.selected_midi_device = ttk.StringVar()
        if self.available_midi_devices:
            self.selected_midi_device.set(self.available_midi_devices[0])

        self.osc_host = ttk.StringVar(value="127.0.0.1")
        self.osc_port = ttk.StringVar(value="9876")
        self.streaming = False
        self.stream_status = ttk.StringVar(value="Inactive")
        self.notification_message = ttk.StringVar()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.build_ui()

    def get_available_cameras(self) -> list[str]:
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(str(i))
                cap.release()
            else:
                break
        return available if available else ["No cameras detected"]

    def get_available_midi_devices(self) -> list[str]:
        try:
            ports = mido.get_input_names()
            return ports if ports else ["No MIDI devices detected"]
        except Exception as e:
            print(f"Error getting MIDI devices: {e}")
            return ["Error detecting MIDI devices"]

    def build_ui(self):
        # Calibration
        calib_frame = ttk.LabelFrame(self.root, text="Calibration", padding=10)
        calib_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(
            calib_frame,
            text="Configure Camera Orientation",
            command=self.run_camera_orientation_calibration,
            bootstyle=PRIMARY,
        ).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(
            calib_frame, text="Configure Keyboard Geometry", command=self.run_keyboard_calibration, bootstyle=PRIMARY
        ).pack(side="left", expand=True, fill="x", padx=5)

        # Camera Settings
        camera_frame = ttk.LabelFrame(self.root, text="Camera Settings", padding=10)
        camera_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(camera_frame, text="Camera index:").grid(row=0, column=0, sticky="w")
        camera_combo = ttk.Combobox(
            camera_frame, values=self.available_cameras, textvariable=self.selected_camera, width=10, bootstyle=INFO
        )
        camera_combo.grid(row=0, column=1, padx=5)
        ttk.Label(
            camera_frame,
            text=(
                f"Found {len(self.available_cameras)} camera(s)"
                if self.available_cameras[0] != "No cameras detected"
                else "No cameras detected"
            ),
        ).grid(row=0, column=2, sticky="w")

        # MIDI Settings
        midi_frame = ttk.LabelFrame(self.root, text="MIDI Settings", padding=10)
        midi_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(midi_frame, text="MIDI input device:").grid(row=0, column=0, sticky="w")
        midi_combo = ttk.Combobox(
            midi_frame,
            values=self.available_midi_devices,
            textvariable=self.selected_midi_device,
            width=25,
            bootstyle=INFO,
        )
        midi_combo.grid(row=0, column=1, padx=5)
        ttk.Label(
            midi_frame,
            text=(
                f"Found {len(self.available_midi_devices)} MIDI device(s)"
                if self.available_midi_devices[0] != "No MIDI devices detected"
                else "No MIDI devices detected"
            ),
        ).grid(row=0, column=2, sticky="w")

        # OSC Settings
        osc_frame = ttk.LabelFrame(self.root, text="OSC Settings", padding=10)
        osc_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(osc_frame, text="OSC Host:").grid(row=0, column=0, sticky="w")
        ttk.Entry(osc_frame, textvariable=self.osc_host, width=15).grid(row=0, column=1, padx=5)
        ttk.Label(osc_frame, text="OSC Port:").grid(row=0, column=2, sticky="w")
        ttk.Entry(osc_frame, textvariable=self.osc_port, width=8).grid(row=0, column=3, padx=5)

        # Stream Control
        stream_frame = ttk.LabelFrame(self.root, text="Stream Control", padding=10)
        stream_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(stream_frame, text="Status:").pack(side="left")
        self.status_label = ttk.Label(stream_frame, textvariable=self.stream_status, bootstyle="danger")
        self.status_label.pack(side="left", padx=5)
        ttk.Button(stream_frame, text="Start Stream", command=self.run_live_runner, bootstyle=SUCCESS, width=20).pack(
            side="right"
        )

        # Notification
        ttk.Label(self.root, textvariable=self.notification_message, bootstyle="danger").pack(
            side="bottom", fill="x", padx=10, pady=5
        )

    def run_camera_orientation_calibration(self):
        try:
            subprocess.Popen(["python", "calibrate_camera_orientation.py"])
        except Exception as e:
            self.notification_message.set(f"Error running camera calibration: {e}")

    def run_keyboard_calibration(self):
        try:
            subprocess.Popen(["python", "calibrate_keyboard.py"])
        except Exception as e:
            self.notification_message.set(f"Error running keyboard calibration: {e}")

    def run_live_runner(self):
        if self.streaming:
            return
        self.notification_message.set("")
        self.streaming = True
        self.stream_status.set("Active")
        self.status_label.config(bootstyle="success")

        try:
            camera_index = self.selected_camera.get()
            midi_device = self.selected_midi_device.get()
            cmd = ["python", "live_runner.py", "--camera", str(camera_index)]
            if midi_device and "No MIDI" not in midi_device and "Error" not in midi_device:
                cmd.extend(["--midi-port", midi_device])
            print(f"Running command: {' '.join(cmd)}")
            self.live_process = subprocess.Popen(cmd)
            self.root.after(1000, self.check_live_process)
        except Exception as e:
            self.notification_message.set(f"Error starting live_runner: {e}")
            self.stop_live_runner()

    def check_live_process(self):
        if hasattr(self, "live_process"):
            assert self.live_process is not None
            retcode = self.live_process.poll()
            if retcode is not None:
                if retcode != 0:
                    self.notification_message.set(f"live_runner exited with code {retcode}")
                self.stop_live_runner()
            else:
                self.root.after(1000, self.check_live_process)

    def stop_live_runner(self):
        if hasattr(self, "live_process") and self.live_process:
            try:
                self.live_process.terminate()
                self.live_process = None
            except Exception as e:
                print(f"Error terminating live_runner: {e}")
        self.streaming = False
        self.stream_status.set("Inactive")
        self.status_label.config(bootstyle="danger")

    def on_closing(self):
        if hasattr(self, "live_process") and self.live_process:
            self.stop_live_runner()
        self.root.destroy()
        import os

        os._exit(0)


def main():
    app = ttk.Window(themename="flatly")
    PianoToolboxApp(app)
    app.mainloop()


if __name__ == "__main__":
    main()
