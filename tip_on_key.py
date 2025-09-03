import cv2
import numpy as np
import utils
import draw_keys_3d
import track_hands


def find_tip_on_key(midi_pitch, note_properties, mp_result, img_output=None):
    hand = note_properties["hand"]
    finger = note_properties["finger"]
    if hand is None or finger is None:
        return None

    tip_idx = track_hands.finger_to_tip_index[finger[0]]
    x_tip = mp_result[hand + "_landmarks_xyz"][0][tip_idx] * track_hands.image_width_px
    y_tip = mp_result[hand + "_landmarks_xyz"][1][tip_idx] * track_hands.image_height_px

    key_outline = draw_keys_3d.pixel_coordinates_of_bounding_box(midi_pitch)
    u, v = point_to_trapezoid_coords((x_tip, y_tip), key_outline)
    if img_output is not None:
        draw_tip_on_key(img_output, key_outline, (x_tip, y_tip), (u, v))

    return u, v


def draw_tip_on_key(
    img, key_bounding_box, tip_xy_coords, tip_uv_coords, show_bb=False, show_text=False
):
    """
    Draw a key and a fingertip point with the trapezoid coordinates.
    """
    # Draw key
    if show_bb:
        pts = np.int32(key_bounding_box).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 255, 255), 1)

    # Draw fingertip as a small square
    square_size = 3  # Half side length of the square
    x, y = int(tip_xy_coords[0]), int(tip_xy_coords[1])
    cv2.rectangle(
        img,
        (x - square_size, y - square_size),  # Top left corner
        (x + square_size, y + square_size),  # Bottom right corner
        (255, 0, 0),  # Color (Blue)
        -1,
    )  # Filled

    # Calculate trapezoid coordinates
    u, v = tip_uv_coords

    # Draw coordinate axes inside the trapezoid
    # Horizontal line at v-coordinate
    p0 = key_bounding_box[0, 0]  # top-left
    p1 = key_bounding_box[1, 0]  # bottom-left
    p3 = key_bounding_box[3, 0]  # top-right
    p2 = key_bounding_box[2, 0]  # bottom-right

    left = p0 + v * (p1 - p0)
    right = p3 + v * (p2 - p3)
    top = p0 + u * (p3 - p0)
    bottom = p1 + u * (p2 - p1)

    # Horizontal line
    start_v = tuple(map(int, left))
    end_v = tuple(map(int, right))
    cv2.line(img, start_v, end_v, (255, 0, 0), 1)

    # Vertical line
    start_h = tuple(map(int, top))
    end_h = tuple(map(int, bottom))
    cv2.line(img, start_h, end_h, (255, 0, 0), 1)

    # Display text with coordinates
    if show_text:
        coord_text = f"u={u:.2f}, v={v:.2f}"
        cv2.putText(
            img,
            coord_text,
            (int(tip_xy_coords[0]) + 10, int(tip_xy_coords[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return img


def point_to_trapezoid_coords(point, trapezoid):
    """
    Calculate the natural coordinates (u,v) of a point inside a trapezoid.

    Args:
        point: Tuple (x,y) for the point to transform
        trapezoid: numpy array with shape (4,1,2) or (4,2) for the trapezoid corners
                  in the order: top-left, bottom-left, bottom-right, top-right

    Returns:
        Tuple (u,v) with values from 0 to 1, where:
        - u=0 is the left side, u=1 is the right side
        - v=0 is the top side, v=1 is the bottom side
    """
    # Extract trapezoid points and convert to easier usable form
    if trapezoid.shape[1] == 1:  # If shape (4,1,2)
        p0 = trapezoid[0, 0]  # top-left
        p3 = trapezoid[1, 0]  # bottom-left
        p2 = trapezoid[2, 0]  # bottom-right
        p1 = trapezoid[3, 0]  # top-right
    else:  # If shape (4,2)
        p0 = trapezoid[0]  # top-left
        p3 = trapezoid[1]  # bottom-left
        p2 = trapezoid[2]  # bottom-right
        p1 = trapezoid[3]  # top-right

    # Iterative approximation of bilinear coordinates
    # Start value in the middle
    u, v = 0.5, 0.5
    max_iterations = 10
    tolerance = 1e-6

    for _ in range(max_iterations):
        # Calculate point at current (u,v)
        top = p0 + u * (p1 - p0)  # Point on top edge
        bottom = p3 + u * (p2 - p3)  # Point on bottom edge
        computed_point = top + v * (bottom - top)

        # Check if we are close enough to the target point
        error = np.linalg.norm(computed_point - point)
        if error < tolerance:
            break

        # Otherwise update u and v
        # Calculate derivatives with respect to u and v
        du_vector = (1 - v) * (p1 - p0) + v * (p2 - p3)
        dv_vector = bottom - top

        # Create Jacobian matrix
        J = np.column_stack([du_vector, dv_vector])

        # Calculate adjustment for u and v
        try:
            delta = np.linalg.solve(J, point - computed_point)
            u += delta[0]
            v += delta[1]
        except np.linalg.LinAlgError:
            # If matrix is singular, exit early
            break

    return u, v


def test_interactive():
    """Interactive test with mouse clicks."""
    draw_keys_3d.init()

    # Load image
    img_path = utils.get_keyboard_image_file_path()
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return

    img = utils.flip_image(img)

    # Draw key (e.g. C4 = MIDI 60)
    midi_pitch = 60

    # Mouse click handler
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            img_copy = img.copy()
            key_outline = draw_keys_3d.pixel_coordinates_of_bounding_box(midi_pitch)
            u, v = point_to_trapezoid_coords((x, y), key_outline)
            draw_tip_on_key(img_copy, key_outline, (x, y), (u, v), show_bb=True)
            cv2.imshow("Test Trapezoid Coordinates", img_copy)

    # Set up window
    cv2.namedWindow("Test Trapezoid Coordinates")
    cv2.setMouseCallback("Test Trapezoid Coordinates", mouse_callback)

    # Initially draw the key
    key_outline = draw_keys_3d.pixel_coordinates_of_bounding_box(midi_pitch)
    pts = np.int32(key_outline).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    cv2.imshow("Test Trapezoid Coordinates", img)
    key_outline = draw_keys_3d.pixel_coordinates_of_bounding_box(midi_pitch)
    key_outline_pts = np.int32(key_outline).reshape((-1, 1, 2))
    cv2.polylines(img, [key_outline_pts], True, (0, 255, 0), 2)
    cv2.imshow("Test Trapezoid Coordinates", img)

    print("Click on the key to see trapezoid coordinates. Press 'q' to quit.")
    while True:
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_interactive()
