import cv2
from ultralytics import YOLO

# Load the YOLOv8 nano model
# Ultralytics will download yolov8n.pt if it is not already available
model = YOLO("yolov8n.pt")

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw detection boxes and labels
    annotated_frame = results[0].plot()

    # Show the result
    cv2.imshow("YOLOv8 Customer Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()