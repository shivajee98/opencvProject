import cv2
import mediapipe as mp
import numpy as np
import subprocess

# Initialize MediaPipe hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a VideoCapture object for webcam
cap = cv2.VideoCapture(0)


# Helper function to change volume using pactl (Ubuntu)
def set_volume(volume_level):
    volume_percentage = f'{volume_level}%'
    subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', volume_percentage])


# Function to calculate distance between two landmarks
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Initialize MediaPipe hands module
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty frame.")
            continue

        # Flip the frame horizontally for selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(rgb_frame)

        # If hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of the thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                h, w, c = frame.shape  # Dimensions of the frame

                # Convert landmarks to pixel positions
                thumb_tip_x = int(thumb_tip.x * w)
                thumb_tip_y = int(thumb_tip.y * h)
                index_finger_tip_x = int(index_finger_tip.x * w)
                index_finger_tip_y = int(index_finger_tip.y * h)

                # Draw circles on the tips
                cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (index_finger_tip_x, index_finger_tip_y), 10, (0, 255, 0), cv2.FILLED)

                # Calculate distance between thumb tip and index finger tip
                distance = calculate_distance(thumb_tip_x, thumb_tip_y, index_finger_tip_x, index_finger_tip_y)

                # Increase sensitivity by lowering the max_distance
                max_distance = 150  # Reduced to make gestures more sensitive
                volume_level = np.clip(int((1 - distance / max_distance) * 200), 0,
                                       200)  # Allowing volume to go up to 200%

                # Set the system volume based on the calculated distance
                set_volume(volume_level)

                # Display the volume level on the frame
                cv2.putText(frame, f'Volume: {volume_level}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Volume Control with Finger Gestures', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
