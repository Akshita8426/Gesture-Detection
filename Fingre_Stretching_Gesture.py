import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for visualizing the results
mp_drawing = mp.solutions.drawing_utils

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize variables for fan speed
fan_speed = 0  # Range from 0 to 100
fan_speed_max = 100
fan_speed_min = 0  # Set a minimum fan speed (0%) to start from zero

# Initialize your desired fan speed intervals
fan_speed_interval = 10  # Change the interval to 10%

# Bar parameters
bar_width = 20
bar_height = 200
bar_position = (50, 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand landmarks
            landmarks = hand_landmarks.landmark

            # Get distances between finger landmarks
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate distances between fingers
            thumb_index_distance = cv2.norm(np.array([thumb_tip.x, thumb_tip.y]), np.array([index_tip.x, index_tip.y]))
            middle_ring_distance = cv2.norm(np.array([middle_tip.x, middle_tip.y]), np.array([ring_tip.x, ring_tip.y]))
            ring_pinky_distance = cv2.norm(np.array([ring_tip.x, ring_tip.y]), np.array([pinky_tip.x, pinky_tip.y]))

            # Calculate the average distance between fingers
            avg_distance = (thumb_index_distance + middle_ring_distance + ring_pinky_distance) / 3

            # Lock the fan speed at specific intervals
            fan_speed = int(round(np.interp(avg_distance, [0, 0.2], [fan_speed_min, fan_speed_max]) / fan_speed_interval) * fan_speed_interval)

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw fan speed bar
    bar_length = int((fan_speed / 100) * bar_height)
    cv2.rectangle(frame, (bar_position[0], bar_position[1] + bar_height - bar_length),
                  (bar_position[0] + bar_width, bar_position[1] + bar_height), (255, 0, 0), -1)

    # Display fan speed on the frame
    cv2.putText(frame, f"Fan Speed: {fan_speed}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Fan Speed Control', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
