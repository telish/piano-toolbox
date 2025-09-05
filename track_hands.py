from typing import Any, TypedDict

import cv2
import mediapipe as mp
import numpy.typing as npt

import osc_sender

MP_THUMB_TIP = 4
MP_INDEX_FINGER_TIP = 8
MP_MIDDLE_FINGER_TIP = 12
MP_RING_FINGER_TIP = 16
MP_PINKY_TIP = 20

finger_to_tip_index = {
    1: MP_THUMB_TIP,
    2: MP_INDEX_FINGER_TIP,
    3: MP_MIDDLE_FINGER_TIP,
    4: MP_RING_FINGER_TIP,
    5: MP_PINKY_TIP,
}

# This is set to actual values once analyze_frame is first called
image_height_px, image_width_px = 0, 0


class TrackingResult(TypedDict):
    left_visible: bool
    right_visible: bool
    left_landmarks_xyz: tuple[list[float], list[float], list[float]] | None
    right_landmarks_xyz: tuple[list[float], list[float], list[float]] | None


def analyze_frame(img_input: npt.NDArray[Any], img_output: npt.NDArray[Any] | None = None) -> TrackingResult:
    global image_height_px, image_width_px
    image_height_px, image_width_px, _ = img_input.shape

    # Convert the frame to RGB (MediaPipe expects RGB images)
    rgb_frame = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    # Process the frame and get results
    mp_results = hands.process(rgb_frame)

    result: TrackingResult = {
        "left_visible": False,
        "right_visible": False,
        "left_landmarks_xyz": None,
        "right_landmarks_xyz": None,
    }
    if mp_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
            # Determine the hand label
            label = mp_results.multi_handedness[idx].classification[0].label
            if label.lower() == "left":
                label = "right"
            elif label.lower() == "right":
                label = "left"
            osc_sender.send_message(f"/{label}/visible", 1)

            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            z_coords = [landmark.z for landmark in hand_landmarks.landmark]

            if label == "left":
                result["left_visible"] = True
                result["left_landmarks_xyz"] = (x_coords, y_coords, z_coords)

            elif label == "right":
                result["right_visible"] = True
                result["right_landmarks_xyz"] = (x_coords, y_coords, z_coords)

            coords = zip(x_coords, y_coords, z_coords)
            flat_coords = []
            for x, y, z in coords:
                flat_coords.append(x)
                flat_coords.append(y)
                flat_coords.append(z)

            if label == "left":
                mp.solutions.drawing_utils.draw_landmarks(
                    img_output,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 200), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2),
                )
            elif label == "right":
                mp.solutions.drawing_utils.draw_landmarks(
                    img_output,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

            osc_sender.send_message(f"/{label}/landmarks", *flat_coords)

    if not result["left_visible"]:
        osc_sender.send_message("/left/visible", 0)
    if not result["right_visible"]:
        osc_sender.send_message("/right/visible", 0)

    return result


hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    osc_sender.configure(9876)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        analyze_frame(img, img)
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the video feed
            break

    cap.release()
    cv2.destroyAllWindows()
