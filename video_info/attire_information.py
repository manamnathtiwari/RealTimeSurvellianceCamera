import cv2
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from facenet_pytorch import MTCNN

# Initialize the age classification model and the face detection model
age_model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
face_detector = MTCNN(keep_all=True)

# Initialize a counter for unique IDs
id_counter = 0

def detect_and_classify_age(frame):
    global id_counter
    # Detect faces in the frame
    boxes, _ = face_detector.detect(frame)
    results = []

    if boxes is not None:
        for box in boxes:
            # Extract each face from the frame
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)

            # Transform the face and pass it through the age model
            inputs = feature_extractor(face_pil, return_tensors='pt')
            output = age_model(**inputs)

            # Get the predicted age
            age = output.logits.softmax(1).argmax(1).item()

            # Assign a unique ID and store the result
            person_id = f"Person_{id_counter}"
            id_counter += 1
            results.append({"id": person_id, "age": age, "box": (x1, y1, x2, y2)})

    return results

# Initialize the webcam (or any video source)
cap = cv2.VideoCapture(0)  # Replace 0 with the path to a video file if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and classify age
    results = detect_and_classify_age(frame)

    # Draw boxes and labels on the frame
    for result in results:
        x1, y1, x2, y2 = result["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{result['id']} Age: {result['age']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Age Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
