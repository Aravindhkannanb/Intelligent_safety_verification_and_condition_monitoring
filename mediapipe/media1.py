import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
falling_frames = 0
threshold_frames = 20  # Adjust as needed


# Open webcam
cap = cv2.VideoCapture(0)

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

        # Check if the person is falling (e.g., if the nose is below ankle level)
        if nose_y > max(left_ankle_y, right_ankle_y):
            falling_frames += 1
            if falling_frames >= threshold_frames:
                print("Fall detected!")
                # Add any additional actions or alerts here
        else:
            falling_frames = 0

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Display the frame
    cv2.imshow("Falling Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
