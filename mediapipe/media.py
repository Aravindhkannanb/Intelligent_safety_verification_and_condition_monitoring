import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
falling_frames = 0
threshold_frames = 20  # Adjust as needed
fall_detection_text = ""
fall_detection_history = deque(maxlen=100)  # Store falling detection status history

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize plot for fall detection history
plt.ion()  # Turn on interactive mode for real-time plotting
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_ylim(0, 1)
ax.set_title("Fall Detection History")
ax.set_xlabel("Frames")
ax.set_ylabel("Fall Detection (1: Detected, 0: Not Detected)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Extract relevant landmarks
        nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
        left_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        # Check if the person is falling based on multiple criteria
        if (
            nose_y > max(left_ankle_y, right_ankle_y) and
            min(left_ankle_y, right_ankle_y) > max(left_shoulder_y, right_shoulder_y)
        ):
            falling_frames += 1
            if falling_frames >= threshold_frames:
                fall_detection_text = "Fall detected!"
                fall_detection_history.append(1)
                # Add any additional actions or alerts here
            else:
                fall_detection_history.append(0)
        else:
            falling_frames = 0
            fall_detection_text = ""
            fall_detection_history.append(0)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Display the fall detection status in the frame
    cv2.putText(frame, fall_detection_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Falling Detection", frame)

    # Update the real-time plot
    line.set_ydata(list(fall_detection_history))
    line.set_xdata(np.arange(len(fall_detection_history)))
    plt.draw()
    plt.pause(0.01)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam, close all windows, and close the plot
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
