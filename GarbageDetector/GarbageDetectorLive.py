import cv2
import math
import cvzone
from ultralytics import YOLO

# Capture video from the laptop's default camera
cap = cv2.VideoCapture(0)  # Use 1 or 2 if you have multiple cameras

# Set desired resolution (optional)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")  # Replace with your actual path

# Define class names
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

while True:
    success, img = cap.read()
    if not success:
        break

    # Get YOLO results
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.1:
                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                                   (max(0, x1), max(35, y1)), 
                                   scale=1, thickness=1)

    # Show result
    cv2.imshow("Garbage Detection - Webcam", img)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy
cap.release()
cv2.destroyAllWindows()
