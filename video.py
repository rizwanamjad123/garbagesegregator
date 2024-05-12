import os
from ultralytics import YOLO
import cv2
import numpy as np
import pyrebase

config = {
    "apiKey": "AIzaSyAx8ahaMapKYwPWiAtGPEMw_oQ6bmnnWh8",
    "authDomain": "waste-segregator-bccf6",
    "databaseURL": "https://waste-segregator-bccf6-default-rtdb.firebaseio.com/",
    "storageBucket": "936087067718"
}
# Initialize Firebase
firebase = pyrebase.initialize_app(config)

# Create a reference to the Firebase database
db = firebase.database()

ProjectBucket = db.child("detection")
# Load a model
model_path = "wastedataset/runs/detect/train/weights/last.pt"
model = YOLO(model_path)  # load a custom model

# Set the confidence threshold (e.g., 0.5 for a 50% confidence threshold)
confidence_threshold = 0.4

# Set the desired width and height for resizing
new_width = 640
new_height = 480

# Open the webcam
cap = cv2.VideoCapture(0)

# Process each frame from the webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there's an issue with the webcam

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Perform object detection on the resized frame
    results = model(frame)[0]

    class_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

    for idx, result in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, confidence, class_id = result

        # Check if the detection confidence is above the threshold
        if confidence >= confidence_threshold:
            # Draw bounding box with green color
            class_color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), class_color, 4)

            # Get class name from class ID
            class_name = class_names[int(class_id)]

            # Write class name text
            text = f"{class_name}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
            print(class_name)

            # Update Firebase database with the detected class name
            db.child("detection").set(class_name)

    # Display the annotated frame
    cv2.imshow('Live Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit the loop

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

