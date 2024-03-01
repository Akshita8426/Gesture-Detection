'''import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert color because MediaPipe uses RGB images
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine if it's left or right hand
                handedness = 'Right' if results.multi_handedness[idx].classification[0].label == 'Right' else 'Left'

                # Change color based on handedness
                color = (0, 255, 0) if handedness == 'Right' else (255, 0, 0)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=color),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=color))

                # Detect thumb up/down gestures
                thumb_tip = hand_landmarks.landmark[4]
                index_finger_tip = hand_landmarks.landmark[8]

                if thumb_tip.y < index_finger_tip.y:
                    status = "Turn On"
                else:
                    status = "Turn Off"

                # Display status on the screen
                cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display accuracy of detection
                accuracy = results.multi_handedness[idx].classification[0].score
                cv2.putText(image, f"Accuracy: {accuracy}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

        # Display the image
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
'''
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for visualizing the results
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for status and accuracy
status = "Off"
accuracy = 0
gesture = "None"
# Initialize VideoCapture
cap = cv2.VideoCapture(0)

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

            # Get thumb and index finger tip coordinates
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Determine left or right hand based on x-coordinate of thumb tip
            hand_color = (255, 0, 0) if thumb_tip.x < index_tip.x else (0, 255, 0)

            # Check if thumb is above the index finger for thumbs up
            if thumb_tip.y < index_tip.y and accuracy > 0.75:
                status = "On"
                gesture = "Thumbs Up"
            else:
                status = "Off"
                gesture = "Thumbs Down"

            # Calculate accuracy based on the vertical distance between thumb and index finger
            accuracy = 1 - abs(thumb_tip.y - index_tip.y)

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4))

    # Display status, detected gesture, and accuracy on the frame
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Thumbs Up/Down Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()


