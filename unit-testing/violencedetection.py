import cv2
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from torchvision import transforms

# Load the VQA model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Define the question
question = "Is there any girl in the frame ?"

# Define a transformation for the image frame 
transform = transforms.ToPILImage()

def ask_question(frame):
    # Convert the OpenCV frame to a PIL image
    pil_image = transform(frame)

    # Prepare inputs for the VQA model
    encoding = processor(pil_image, question, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()

    # Get the predicted answer
    predicted_answer = model.config.id2label[idx]
    
    # Return True if harassment is detected, otherwise False
    if "yes" in predicted_answer.lower():
        return True
    return False

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process the frame to ask if harassment is detected
    harassment_detected = ask_question(frame)

    # Show the result on the frame
    if harassment_detected:
        cv2.putText(frame, "Harassment Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print(" True")
    else:
        cv2.putText(frame, "No Harassment", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(" False")

    # Display the frame
    cv2.imshow("Harassment Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
