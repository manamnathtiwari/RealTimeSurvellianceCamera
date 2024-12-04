import cv2
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import numpy as np

# Initialize model and processor with error handling
try:
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define questions for harassment detection
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

def detect_harassment(frame):
    """Detects potential harassment in the given frame and returns a normalized harassment score."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    harassment_score = 0
    for question in questions:
        try:
            encoding = processor(pil_image, question, return_tensors="pt")
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = model.config.id2label[idx]
            if answer.lower() == "yes":
                harassment_score += 1
        except Exception as e:
            print(f"Error during question processing: {e}")
            continue

    # Normalize score by the number of questions
    normalized_score = harassment_score / len(questions)
    return normalized_score
