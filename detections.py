import os
from ultralytics import YOLO
import cv2
import numpy as np

# Define the path to your image and output image
image_path = 'wastedataset/test/images/metal4_jpg.rf.95bf1c428ead5f8f591d5597fd3f81a6.jpg'

# Load a model
model_path = "wastedataset/runs/detect/train/weights/last.pt"
model = YOLO(model_path)  # load a custom model

# Set the confidence threshold (e.g., 0.5 for a 50% confidence threshold)
confidence_threshold = 0.5

new_width = 640
new_height = 640

# Load the image
frame = cv2.imread(image_path)

# Resize the image
resized_image = cv2.resize(frame, (new_width, new_height))

results = model(resized_image)[0]

# Set color to green (0, 255, 0)
class_color = (0, 255, 0)

class_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

for idx, result in enumerate(results.boxes.data.tolist()):
    x1, y1, x2, y2, confidence, class_id = result

    # Check if the detection confidence is above the threshold
    if confidence >= confidence_threshold:
        # Draw bounding box with green color
        cv2.rectangle(resized_image, (int(x1), int(y1)), (int(x2), int(y2)), class_color, 4)

        # Get class name from class ID
        class_name = class_names[int(class_id)]

        # Write class name text
        text = f"{class_name}"
        cv2.putText(resized_image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)

# Display the annotated image (optional)
cv2.imshow('Disease Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
