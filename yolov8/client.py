import cv2
import socket
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('safety_gadgets.pt')

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the server address and port
server_address = ('192.168.116.139', 8888)  # Replace <RASPBERRY_PI_IP> with the Raspberry Pi's IP address

# Connect to the server
client_socket.connect(server_address)

# Open the video file
video_path = "project.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, conf=0.8)
        output = "Safe"  # Default to safe

        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                output = "No"

        # Send the safety status to the Raspberry Pi
        client_socket.send(output.encode())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the connection
cap.release()
client_socket.close()
