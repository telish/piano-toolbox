import cv2
import mediapipe as mp
import json

from pythonosc import udp_client

MP_THUMB_TIP = 4
MP_INDEX_FINGER_TIP = 8
MP_MIDDLE_FINGER_TIP = 12
MP_RING_FINGER_TIP = 16
MP_PINKY_TIP = 20


def analyze_frame(frame):
    # Flip the frame for a more natural view
    # frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB (MediaPipe expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get results
    mp_results = hands.process(rgb_frame)

    result = {
        "left_visible": False,
        "right_visible": False,
        "left_landmarks_xyz": None,
        "right_landmarks_xyz": None
    }
    if mp_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
            # Determine the hand label
            label = mp_results.multi_handedness[idx].classification[0].label
            udp_client.send_message(f"/{label.lower()}/visible", 1)

            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            z_coords = [landmark.z for landmark in hand_landmarks.landmark]

            if label.lower() == "left":
                result["left_visible"] = True
                result["left_landmarks_xyz"] = (x_coords, y_coords, z_coords)

            elif label.lower() == "right":
                result["right_visible"] = True
                result["right_landmarks_xyz"] = (x_coords, y_coords, z_coords)

            coords = zip(x_coords, y_coords, z_coords)
            flat_coords = []
            for x, y, z in coords:
                flat_coords.append(x)
                flat_coords.append(y)
                flat_coords.append(z)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks,  mp.solutions.hands.HAND_CONNECTIONS)
            udp_client.send_message(f"/{label.lower()}/landmarks", flat_coords)

    if not result["left_visible"]:
        udp_client.send_message(f"/left/visible", 0)
    if not result["right_visible"]:
        udp_client.send_message(f"/right/visible", 0)

    return frame, result


with open('config.json', 'r') as file:
    config = json.load(file)
config = config['track_hands.py']
udp_client = udp_client.SimpleUDPClient("127.0.0.1", config["port_outgoing"])

hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 for Laptop camera. 1 for Elgato

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img, _ = analyze_frame(frame)
        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video feed
            break

    cap.release()
    cv2.destroyAllWindows()
