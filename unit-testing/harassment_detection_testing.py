import cv2   
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import numpy as np
from collections import deque
import time

# Initialize model and processor with error handling
try:
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Questions for harassment detection
questions = [
    "Is there any harassment occurring in the frame?",
    "Is there any physical assault taking place in the frame?",
    "Is there a woman or child showing signs of distress or discomfort?",
    "Is there any unwanted or inappropriate physical contact in the frame?",
    "Is someone displaying threatening or aggressive body language toward another person?",
    "Is there any visible attempt to intimidate or control someone?",
    "Are there signs of a person being physically restrained or blocked from leaving?",
    "Is there a group surrounding or closely approaching an individual in a threatening way?",
    "Is anyone trying to escape or move away from another person in the frame?",
    "Is there visible verbal aggression, such as yelling or pointing aggressively?",
    "Are there gestures or actions that suggest emotional abuse or mocking?",
    "Is there any action indicating unwanted proximity toward a woman or child?",
    "Are there expressions of anger or raised hands in a threatening manner?",
    "Does the frame show any attempt to corner or trap someone?",
    "Is there a person who appears visibly uncomfortable or defensive?",
    "Are there inappropriate or aggressive gestures aimed at a woman or child?",
    "Does the frame capture an interaction where someone appears to be in fear?",
    "Is there a minor (child) showing signs of distress or avoidance behavior?",
    "Are there any confrontational or forceful gestures directed at a specific person?",
    "Is there a woman being subjected to unwanted attention or inappropriate behavior?"
]

# Initialize video capture (set video file path here)
cap = cv2.VideoCapture("C:/Users/Manamnath tiwari/Downloads/Ep 4_ Sexual Harassment at Workplace.mp4")  # Replace with 0 for webcam, or 'your_video_file.mp4' for a file

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set parameters
frame_count = 0
window_size = 30  # seconds
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Use default if FPS is 0
frame_buffer = deque(maxlen=int(window_size * fps))  # Buffer to store harassment scores
start_time = time.time()

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video file or frame capture failed, ending video processing.")
        break

    # Resize for display
    frame = cv2.resize(frame, (400, 500))

    # Convert frame to PIL image for processing
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    harassment_score = 0
    question_answers = []
    
    try:
        for question in questions:
            # Model processing
            encoding = processor(pil_image, question, return_tensors="pt")
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            
            # Interpreting model answer
            answer = model.config.id2label[idx]
            question_answers.append((question, answer))
            if answer.lower() == "yes":
                harassment_score += 1
    except Exception as e:
        print(f"Error during question processing: {e}")
        break

    # Calculate normalized score
    normalized_score = harassment_score / len(questions)
    frame_buffer.append(normalized_score)

    # Output to terminal instead of display
    print(f"Frame {frame_count}")
    for question, answer in question_answers:
        print(f"{question}: {answer}")
    print(f"Harassment Score: {normalized_score:.2f}\n")

    # Periodic check for harassment detection
    if time.time() - start_time >= window_size:
        avg_harassment_score = np.mean(frame_buffer)
        print("\n=== Harassment Detection Summary ===")
        if avg_harassment_score > 0.5:
            print("Harassment Detected in the last 30 seconds\n")
        else:
            print("No Harassment Detected in the last 30 seconds\n")
        start_time = time.time()

    # Show the frame
    cv2.imshow("Harassment Detection", frame)
    frame_count += 1

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
