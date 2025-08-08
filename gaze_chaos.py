import cv2
import numpy as np
import mediapipe as mp
import time
import random
import string
from pynput import keyboard, mouse
import collections

# --- CONFIGURATION ---
# Gaze detection sensitivity. Lower numbers mean you have to look further away to trigger the effect.
# A good range is 0.08 to 0.2
GAZE_THRESHOLD_X = 0.1
GAZE_THRESHOLD_Y = 0.15
CALIBRATION_TIME_SECONDS = 3

# --- GLOBAL STATE VARIABLES ---
keyboard_should_scramble = False
mouse_should_invert = False
last_mouse_pos = (0, 0)
is_inverting_flag = False # Prevents recursive mouse events
calibrated_center_x = 0.5
calibrated_center_y = 0.5
is_calibrated = False

# --- INITIALIZE MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # This is crucial for iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- KEYBOARD CONTROL ---
keyboard_controller = keyboard.Controller()

def get_random_char():
    """Returns a random lowercase letter or number."""
    return random.choice(string.ascii_lowercase + string.digits)

def on_press(key):
    """Callback function for when a key is pressed."""
    global keyboard_should_scramble
    if keyboard_should_scramble:
        try:
            if hasattr(key, 'char') and key.char and key.char.isalnum():
                random_char = get_random_char()
                keyboard_controller.type(random_char)
                return False # Suppress the original key press
        except AttributeError:
            pass # Special keys (like Shift, Ctrl) will pass through

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

# --- MOUSE CONTROL ---
mouse_controller = mouse.Controller()

def on_move(x, y):
    """Callback function for when the mouse moves."""
    global mouse_should_invert, last_mouse_pos, is_inverting_flag

    if is_inverting_flag:
        return

    if mouse_should_invert:
        dx = x - last_mouse_pos[0]
        dy = y - last_mouse_pos[1]
        
        inverted_x = last_mouse_pos[0] - dx
        inverted_y = last_mouse_pos[1] - dy
        
        is_inverting_flag = True
        mouse_controller.position = (inverted_x, inverted_y)
        is_inverting_flag = False
        
        last_mouse_pos = (inverted_x, inverted_y)
    else:
        last_mouse_pos = (x, y)

mouse_listener = mouse.Listener(on_move=on_move)
mouse_listener.start()

# --- MAIN COMPUTER VISION LOOP ---
def main():
    global keyboard_should_scramble, mouse_should_invert, calibrated_center_x, calibrated_center_y, is_calibrated
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Starting Gaze-Controlled Chaos...")
    print("Press 'q' in the OpenCV window to quit.")
    
    start_time = time.time()
    gaze_history_x = collections.deque(maxlen=30)
    gaze_history_y = collections.deque(maxlen=30)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # --- Gaze Calculation ---
            # Using the center of the left and right iris for stability
            left_iris_center = np.array([face_landmarks[468].x, face_landmarks[468].y])
            right_iris_center = np.array([face_landmarks[473].x, face_landmarks[473].y])
            
            # A simple way to get a gaze ratio is to compare iris position to a face anchor
            # We'll use the nose tip as a rough anchor point
            nose_tip = np.array([face_landmarks[1].x, face_landmarks[1].y])
            
            # Calculate gaze vector relative to the nose
            gaze_vector = (left_iris_center + right_iris_center) / 2 - nose_tip
            
            # Normalize the vector to get a ratio (this is a simplified approach)
            # We add a small epsilon to avoid division by zero
            gaze_ratio_x = gaze_vector[0] * 10 # Multiply to make the value more sensitive
            gaze_ratio_y = gaze_vector[1] * 10

            # --- Calibration Phase ---
            if not is_calibrated:
                elapsed_time = time.time() - start_time
                if elapsed_time < CALIBRATION_TIME_SECONDS:
                    gaze_history_x.append(gaze_ratio_x)
                    gaze_history_y.append(gaze_ratio_y)
                    calib_text = f"Calibrating... Look at camera: {int(CALIBRATION_TIME_SECONDS - elapsed_time)}s"
                    cv2.putText(image, calib_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    if len(gaze_history_x) > 0:
                        calibrated_center_x = sum(gaze_history_x) / len(gaze_history_x)
                        calibrated_center_y = sum(gaze_history_y) / len(gaze_history_y)
                        is_calibrated = True
                        print(f"Calibration complete. Center X: {calibrated_center_x:.2f}, Center Y: {calibrated_center_y:.2f}")
                    else:
                        print("Calibration failed. No face detected. Using defaults.")
                        is_calibrated = True # Fail gracefully

            # --- Main Logic Phase ---
            if is_calibrated:
                # Check if gaze is outside the calibrated "safe zone"
                if (abs(gaze_ratio_x - calibrated_center_x) > GAZE_THRESHOLD_X or
                    abs(gaze_ratio_y - calibrated_center_y) > GAZE_THRESHOLD_Y):
                    # LOOKING AWAY
                    keyboard_should_scramble = True
                    mouse_should_invert = False
                    status_text = "MODE: KEYBOARD CHAOS"
                    status_color = (0, 0, 255) # Red
                else:
                    # LOOKING AT SCREEN
                    keyboard_should_scramble = False
                    mouse_should_invert = True
                    status_text = "MODE: MOUSE INVERSION"
                    status_color = (0, 255, 0) # Green
        else:
            # NO FACE DETECTED
            keyboard_should_scramble = False
            mouse_should_invert = False
            status_text = "MODE: NORMAL (No face detected)"
            status_color = (255, 255, 255) # White

        # --- Display Visual Feedback ---
        if is_calibrated:
            cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        cv2.imshow('Gaze-Controlled Chaos', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("Stopping listeners and cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    keyboard_listener.stop()
    mouse_listener.stop()
    face_mesh.close()

if __name__ == '__main__':
    main()
