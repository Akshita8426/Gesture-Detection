from ultralytics import YOLO
import cv2
import math
import cvzone
import matplotlib.pyplot as plt
import mediapipe as mp
cap = cv2.VideoCapture(0)
cap.set(4, 1280)
cap.set(3, 720)
model = YOLO("Wieghts/yolov8n.pt")

#detect person only
'''
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
print("The model use to train the data is: ", model)
confidence_levels = []
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for i in results:
        boxes = i.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # Check if the detected object is a person (class 0)
            if cls == 0:
                conf = math.ceil(box.conf[0] * 100) / 100
                confidence_levels.append(conf)
                print(f"Detected: {classNames[cls]}, Confidence Level: {conf}")
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = (x2 - x1), (y2 - y1)
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f"{classNames[cls]}{conf}", (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Graph of confidance Level
plt.plot(confidence_levels)
plt.xlabel('Frame')
plt.ylabel('Confidence Level')
plt.title('Confidence Level of Human Faces Over Time')
plt.show()
plt.subplot(1, 2, 1)
plt.hist(confidence_levels, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Confidence Level')
plt.ylabel('Frequency')
plt.title('Histogram of Confidence Levels for Detected Persons')

plt.subplot(1, 2, 2)
plt.plot(confidence_levels, label='Person Confidence Level', color='green')
plt.xlabel('Frame')
plt.ylabel('Confidence Level')
plt.title('Real-time Confidence Level Trend for Detected Persons')
plt.legend()
plt.tight_layout()
plt.show()
print("************************** You have Quited by pressing q **************************")
cv2.destroyAllWindows()'''

#Detect All oject
'''

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
print("The model used to train the data is: ", model)

# Dictionary to store confidence levels for each class
confidence_dict = {class_name: [] for class_name in classNames}

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for i in results:
        boxes = i.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = math.ceil(box.conf[0] * 100) / 100

            # Store confidence level for each class in the dictionary
            confidence_dict[classNames[cls]].append(conf)

            print(f"Detected: {classNames[cls]}, Confidence Level: {conf}")

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2 - x1), (y2 - y1)
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f"{classNames[cls]}{conf}", (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    # Display the image
    cv2.imshow("Image", img)

    # Check for user input to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Plot confidence levels for each class
for class_name, confidences in confidence_dict.items():
    plt.plot(confidences, label=class_name)

plt.xlabel('Frame')
plt.ylabel('Confidence Level')
plt.title('Confidence Level of Various Objects Over Time')
plt.legend(loc='upper right')
plt.show()

# Close the OpenCV windows
cv2.destroyAllWindows()
'''

#Detect with fingure
'''
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

# Initialize Mediapipe Hands
hands = mp.solutions.hands
mp_hands = hands.Hands()
mp_drawing = mp.solutions.drawing_utils

selected_object = None

def send_signal_to_phone(box):
    # Implement your logic to send a signal to the phone
    print("Sending signal to the phone:", box)

def draw_landmarks(frame, hand_landmarks):
    
    mp_drawing.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)


while True:
    ret, frame = cap.read()

    # Object detection using YOLO
    results = model(frame, stream=True)
    for i in results:
        boxes = i.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Check if the detected object is a cell phone
            if classNames[cls] == "cell phone" and conf > 0.5:
                selected_object = box
                break

    # Hand tracking using MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = mp_hands.process(frame_rgb)

    # If hands are detected, draw landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

    # Draw bounding box around the cell phone
    if selected_object is not None:
        x, y, w, h = selected_object.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object and Hand Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()'''

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

def send_signal_to_phone(box):
    # Implement your logic to send a signal to the phone
    print("Sending signal to the phone:", box)

def draw_landmarks(frame, hand_landmarks):
    # Draw hand landmarks on the frame
    mp_drawing.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)

while True:
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
            if classNames[cls] == "cell phone" and conf > 0.5:
                selected_object = box
                break

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
                    finger_in_box = True
                else:
                    finger_in_box = False

    # Draw bounding box around the cell phone and change color based on finger position
    if selected_object is not None:
        x, y, w, h = selected_object.xyxy[0].cpu().numpy().astype(int)
        color = (0, 255, 0) if not finger_in_box else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (w, h), color, 2)

    # Display the frame
    cv2.imshow("Object and Hand Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()