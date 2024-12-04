from transformers import ViltProcessor, ViltForQuestionAnswering
import cv2
from PIL import Image
import torch

# Initialize the model and processor
processor = ViltProcessor.from_pretrained("MBZUAI/Video-ChatGPT-7B")
model = ViltForQuestionAnswering.from_pretrained("MBZUAI/Video-ChatGPT-7B")

# Define the question to determine age
text = "What is the age of the person?"

# Initialize face detector (using Haar cascades here)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if the classifier was loaded properly
if face_cascade.empty():
    print("Error: Haar Cascade Classifier not loaded!")
    exit()

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize ID for each person
    person_id = 1
    for (x, y, w, h) in faces:
        # Crop face from frame
        face = frame[y:y+h, x:x+w]
        pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Prepare inputs for the model
        encoding = processor(pil_face, text, return_tensors="pt")

        # Perform a forward pass to get the output
        with torch.no_grad():
            outputs = model(**encoding)
        
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        # Get the predicted age
        predicted_age = model.config.id2label[idx]
        print(f"Person ID {person_id}: Predicted age - {predicted_age}")

        # Draw a rectangle around the face with the person ID
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {person_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Increment person ID for the next detected face
        person_id += 1

    # Display the resulting frame
    cv2.imshow('Webcam - Age Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
