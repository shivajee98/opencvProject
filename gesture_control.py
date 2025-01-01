import cv2
import mediapipe as mp
import requests

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define Flask app URL
FLASK_APP_URL = 'http://192.168.233.120:80/send_command/'

# Function to send car control command to Flask app
def send_command_to_car(direction):
    url = FLASK_APP_URL + direction
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Sent command: {direction}")
        else:
            print(f"Failed to send command: {direction}")
    except Exception as e:
        print(f"Error sending command: {e}")

# Function to detect if a specific finger is raised (threshold based)
def is_finger_raised(hand_landmarks, finger_tip, finger_mcp):
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_mcp].y

# Function to detect if the hand is a fist (all fingers closed)
def is_fist(hand_landmarks):
    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    return all(hand_landmarks.landmark[tip].y > 0.7 for tip in tips)

# Define the OpenCV video stream (webcam)
cap = cv2.VideoCapture(0)

# Initialize direction text to display on frame
direction_text = ""

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check each finger status
            index_finger_up = is_finger_raised(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)
            middle_finger_up = is_finger_raised(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
            ring_finger_up = is_finger_raised(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP)
            pinky_finger_up = is_finger_raised(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)

            # Detect which finger is up and send the corresponding command
            if index_finger_up and not middle_finger_up:
                direction_text = "Forward"
                send_command_to_car('F')  # Forward (index finger up)
            elif middle_finger_up and not index_finger_up:
                direction_text = "Backward"
                send_command_to_car('B')  # Backward (middle finger up)
            elif ring_finger_up:
                direction_text = "Right"
                send_command_to_car('R')  # Right (ring finger up)
            elif pinky_finger_up:
                direction_text = "Left"
                send_command_to_car('L')  # Left (pinky finger up)
            elif is_fist(hand_landmarks):
                direction_text = "Stop"
                send_command_to_car('S')  # Stop (fist closed)

    # Draw the detected direction on the frame
    cv2.putText(image, f"Direction: {direction_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Hand Gesture Control', image)

    # Exit the loop with 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
