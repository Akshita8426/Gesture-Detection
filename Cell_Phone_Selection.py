from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3, 1980)
cap.set(4, 1080)


model = YOLO("Weights/yolov8x.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
hands = mp.solutions.hands
mp_hands = hands.Hands()
mp_drawing = mp.solutions.drawing_utils

selected_object = None
finger_in_box = False

# Initialize variables for plotting
timestamps = []
detection_times = []
speeds = []
confidence_levels = []

def send_signal_to_phone(box):
    # Implement your logic to send a signal to the phone
    print("Sending signal to the phone:", box)

def draw_landmarks(frame, hand_landmarks):
    # Draw hand landmarks on the frame
    mp_drawing.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)

# Initialize a variable to track the previous state
finger_was_outside = True

while True:
    start_time = datetime.now()

    ret, frame = cap.read()

    # Object detection using YOLO
    results = model(frame, stream=True)
    selected_object = None  # Reset selected_object

    for i in results:
        boxes = i.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Check if the detected object is a cell phone
            if classNames[cls] == "cell phone" and conf >= 0.3:
                selected_object = box
                break

    # Calculate time taken for detection
    end_time = datetime.now()
    detection_time = (end_time - start_time).total_seconds()

    # Hand tracking using MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = mp_hands.process(frame_rgb)

    # If hands are detected, draw landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            # Check if finger is in the box around the cell phone
            if selected_object is not None:
                x, y, w, h = selected_object.xyxy[0].cpu().numpy().astype(int)
                finger_x, finger_y = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])

                if x < finger_x < w and y < finger_y < h:
                    if finger_was_outside:  # Only change the state if the finger was previously outside the box
                        finger_in_box = not finger_in_box
                        finger_was_outside = False
                else:
                    finger_was_outside = True  # Set the flag to True when the finger is outside the box

    # Draw bounding box around the cell phone and change color based on finger position
    if selected_object is not None:
        x, y, w, h = selected_object.xyxy[0].cpu().numpy().astype(int)
        color = (0, 255, 0) if finger_in_box else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (w, h), color, 2)

    # Display the frame
    cv2.imshow("Object and Hand Tracking", frame)

    # Store data for plotting
    timestamps.append(datetime.now())
    detection_times.append(detection_time)
    speeds.append(0)  # Replace with your calculation for speed
    confidence_levels.append(selected_object.conf[0].item() if selected_object else 0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# Plotting the graph for Detection Time
plt.figure(figsize=(10, 5))
plt.plot(timestamps, detection_times, label='Detection Time (s)')
plt.xlabel('Timestamp')
plt.title('Detection Time')
plt.legend()
plt.show()

# Plotting the graph for Speed
plt.figure(figsize=(10, 5))
plt.plot(timestamps, speeds, label='Speed')
plt.xlabel('Timestamp')
plt.title('Speed')
plt.legend()
plt.show()

# Plotting the graph for Confidence Level
plt.figure(figsize=(10, 5))
plt.plot(timestamps, confidence_levels, label='Confidence Level')
plt.xlabel('Timestamp')
plt.title('Confidence Level')
plt.legend()
plt.show()
