import utils

# Based on: https://upload.wikimedia.org/wikipedia/commons/4/48/Pianoteilung.svg
# Article: https://de.wikipedia.org/wiki/Klaviatur

black_height = 100.  # If you change this value, you have to re_init()
BLACK_WIDTH = 12.7
C_TOP_WIDTH = 15.05
D_TOP_WIDTH = 15.3
E_TOP_WIDTH = 15.05
F_TOP_WIDTH = 13.95
G_TOP_WIDTH = 14.2
A_TOP_WIDTH = 14.2
B_TOP_WIDTH = 13.95
LOWER_A_TOP_WIDTH = A_TOP_WIDTH + BLACK_WIDTH / 2.
WHITE_BOTTOM_WIDTH = 23.6

KEYBOARD_WIDTH = 52 * WHITE_BOTTOM_WIDTH
WHITE_HEIGHT = 145.

white_keys = [
    21, 23, 24, 26, 28, 29, 31,  # lowest octave
    33, 35, 36, 38, 40, 41, 43,  # second octave
    45, 47, 48, 50, 52, 53, 55,  # third octave
    57, 59, 60, 62, 64, 65, 67,  # fourth octave
    69, 71, 72, 74, 76, 77, 79,  # fifth octave
    81, 83, 84, 86, 88, 89, 91,  # sixth octave
    93, 95, 96, 98, 100, 101, 103,  # seventh octave
    105, 107, 108  # highest octave
]
black_keys = [key for key in range(21, 108) if key not in white_keys]

keyboard_outline = {
    "top-left": [0,  0],
    "top-right": [KEYBOARD_WIDTH,  0],
    "bottom-right": [KEYBOARD_WIDTH, WHITE_HEIGHT],
    "bottom-left": [0,  WHITE_HEIGHT]
}


def pitch_class(midi_pitch):
    c = midi_pitch % 12
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E',
                     'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return pitch_classes[c]


def re_init():
    global left_at_bottom, left_at_top, right_at_bottom, right_at_top
    left_at_top = [0]
    for pitch in range(21, 108):
        if pitch == 21:
            left_at_top.append(left_at_top[-1] + LOWER_A_TOP_WIDTH)
        else:
            match pitch_class(pitch):
                case 'C':
                    left_at_top.append(left_at_top[-1] + C_TOP_WIDTH)
                case 'C#':
                    left_at_top.append(left_at_top[-1] + BLACK_WIDTH)
                case 'D':
                    left_at_top.append(left_at_top[-1] + D_TOP_WIDTH)
                case 'D#':
                    left_at_top.append(left_at_top[-1] + BLACK_WIDTH)
                case 'E':
                    left_at_top.append(left_at_top[-1] + E_TOP_WIDTH)
                case 'F':
                    left_at_top.append(left_at_top[-1] + F_TOP_WIDTH)
                case 'F#':
                    left_at_top.append(left_at_top[-1] + BLACK_WIDTH)
                case 'G':
                    left_at_top.append(left_at_top[-1] + G_TOP_WIDTH)
                case 'G#':
                    left_at_top.append(left_at_top[-1] + BLACK_WIDTH)
                case 'A':
                    left_at_top.append(left_at_top[-1] + A_TOP_WIDTH)
                case 'A#':
                    left_at_top.append(left_at_top[-1] + BLACK_WIDTH)
                case 'B':
                    left_at_top.append(left_at_top[-1] + B_TOP_WIDTH)

    left_at_bottom = []
    for i, pitch in enumerate(range(21, 109)):
        if pitch in white_keys:
            white_idx = white_keys.index(pitch)
            left_at_bottom.append(white_idx * WHITE_BOTTOM_WIDTH)
        else:
            left_at_bottom.append(left_at_top[i])

    right_at_top = left_at_top.copy()
    for i, pitch in enumerate(range(21, 109)):
        if pitch == 21:
            right_at_top[i] += LOWER_A_TOP_WIDTH
        elif pitch == 108:
            right_at_top[i] += WHITE_BOTTOM_WIDTH
        elif pitch not in white_keys:
            right_at_top[i] += BLACK_WIDTH
        else:
            match pitch_class(pitch):
                case 'C':
                    right_at_top[i] += C_TOP_WIDTH
                case 'D':
                    right_at_top[i] += D_TOP_WIDTH
                case 'E':
                    right_at_top[i] += E_TOP_WIDTH
                case 'F':
                    right_at_top[i] += F_TOP_WIDTH
                case 'G':
                    right_at_top[i] += G_TOP_WIDTH
                case 'A':
                    right_at_top[i] += A_TOP_WIDTH
                case 'B':
                    right_at_top[i] += B_TOP_WIDTH

    right_at_bottom = left_at_bottom.copy()
    for i, pitch in enumerate(range(21, 109)):
        if pitch in white_keys:
            right_at_bottom[i] += WHITE_BOTTOM_WIDTH
        else:
            right_at_bottom[i] += BLACK_WIDTH

    assert len(left_at_top) == 88
    assert len(left_at_bottom) == 88
    assert len(right_at_top) == 88
    assert len(right_at_bottom) == 88


def key_points(midi_pitch):
    idx = midi_pitch - 21
    if midi_pitch in white_keys:
        return [
            [left_at_top[idx], 0],  # top-left
            [left_at_top[idx], black_height],  # left-middle 1/2
            [left_at_bottom[idx], black_height],  # left-middle 2/2
            [left_at_bottom[idx], WHITE_HEIGHT],  # bottom-left
            [right_at_bottom[idx], WHITE_HEIGHT],  # bottom-right
            [right_at_bottom[idx], black_height],  # right-middle 1/2
            [right_at_top[idx], black_height],  # right-middle 2/2
            [right_at_top[idx], 0]  # top-right
        ]
    else:
        return [
            [left_at_top[idx], 0],  # top-left
            [left_at_top[idx], black_height],  # left-middle
            [right_at_top[idx], black_height],  # right-middle
            [right_at_top[idx], 0]  # top-right
        ]


def key_bounding_box(midi_pitch):
    if midi_pitch not in white_keys:
        return key_points(midi_pitch)
    else:
        idx = white_keys.index(midi_pitch)
        left = idx * WHITE_BOTTOM_WIDTH
        right = left + WHITE_BOTTOM_WIDTH
        return [
            [left, 0],  # top-left
            [left, WHITE_HEIGHT],  # bottom-left
            [right, WHITE_HEIGHT],  # bottom-right
            [right, 0]  # top-right
        ]


re_init()  # Call again, if black_height is changed

if __name__ == "__main__":
    import cv2
    import numpy as np
    import utils

    # Use the keyboard path from utils instead of hardcoded path
    image_path = utils.get_keyboard_image_path()
    img = cv2.imread(image_path)
    img = utils.flip_image(img)

    # Draw keys and labels
    for i, pitch in enumerate(range(21, 109)):
        # Get points and add y-offset
        key_pts = np.array([key_points(pitch)], dtype=np.int32)

        cv2.polylines(img, key_pts, isClosed=True,
                      color=(0, 200, 0), thickness=2)

        # Also shift text
        cv2.putText(img, f'{pitch}', (int(left_at_top[i] + 1), 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Also shift bounding box
        box_pts = np.array([key_bounding_box(pitch)], dtype=np.int32)
        box_pts[0, :, 1] += int(WHITE_HEIGHT + 2)

        cv2.polylines(img, box_pts, isClosed=True,
                      color=(200, 0, 0), thickness=2)

    cv2.imshow("Keyboard Geometry", img)
    cv2.waitKey(0)
