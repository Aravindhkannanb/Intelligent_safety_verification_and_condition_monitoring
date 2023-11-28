import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage
from flask import Flask, render_template, request, redirect, url_for
import os
import pyttsx3
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize Firebase Admin SDK
cred = credentials.Certificate("private.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'facial-recognition-943e2.appspot.com'})

# Initialize Flask
app = Flask(__name__)

# Function to download reference images from Firebase Storage and get encodings
def load_reference_images_from_firebase():
    reference_encodings = {}
    # Reference to Firebase Storage bucket
    bucket = storage.bucket()

    # List all files in the 'reference' folder in Firebase Storage
    blobs = bucket.list_blobs()

    for blob in blobs:
        person_name = os.path.splitext(os.path.basename(blob.name))[0]
        image_path = f"reference/{person_name}.jpg"  # Updated: Use relative path

        # Download image from Firebase Storage
        blob.download_to_filename(image_path)

        # Load the downloaded image and get face encoding
        reference_image = face_recognition.load_image_file(image_path)
        reference_encoding = get_face_encoding(reference_image)

        if reference_encoding is not None:
            reference_encodings[person_name] = reference_encoding

    return reference_encodings

# Function to get face encoding (with error handling)
def get_face_encoding(image):
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        return face_encodings[0]
    else:
        print("No face found in the image.")
        return None

# Function to speak the verification result
def speak_verification_result(matched_person):
    if matched_person != "Unknown":
        engine.say(f"Welcome, {matched_person}!")
    else:
        engine.say("Verification failed. Unknown person.")
    engine.runAndWait()

# Step 1: Load reference images and encodings from Firebase
reference_encodings = load_reference_images_from_firebase()

# Variable to indicate if verification loop should run
verification_loop_running = False

# Video capture thread function
def video_capture_loop():
    global verification_loop_running

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while verification_loop_running:
        # Capture video frame-by-frame
        ret, frame = video_capture.read()

        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the current face with all reference faces
            matched_person = "Unknown"

            for person_name, reference_encoding in reference_encodings.items():
                matches = face_recognition.compare_faces([reference_encoding], face_encoding)
                if matches[0]:
                    matched_person = person_name
                    break

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, matched_person, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Speak the verification result
            speak_verification_result(matched_person)

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# Route for the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        employee_name = request.form['name']
        image_file = request.files['image']

        # Save the image to the 'reference' folder
        image_path = f"reference/{employee_name}.jpg"
        image_file.save(image_path)

        # Load the image and get face encoding
        reference_image = face_recognition.load_image_file(image_path)
        reference_encoding = get_face_encoding(reference_image)

        if reference_encoding is not None:
            reference_encodings[employee_name] = reference_encoding

        return redirect(url_for('register'))

    return render_template('register.html')

# Route for the verification page
# Route for the verification page
@app.route('/verify', methods=['GET', 'POST'])
def verify():
    global verification_loop_running

    if request.method == 'POST':
        # Set verification_loop_running to True before starting the video capture loop
        verification_loop_running = True

        # Start the video capture loop in a separate thread
        video_capture_thread = threading.Thread(target=video_capture_loop)
        video_capture_thread.start()

    return render_template('verify.html')

if __name__ == '__main__':
    # Run the Flask app on localhost:5000
    app.run(debug=True)
