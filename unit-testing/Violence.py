import os  
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque

class ViolenceDetector:
    def __init__(self):
        # Use os.path.join for better path handling (optional)
        model_path = os.path.join(r"C:\Users\Manamnath tiwari\OneDrive\Desktop\Mirage-Real Time Violence Detection", "violence_detection_model12.keras")
        
        # Load the violence detection model
        self.violence_model = load_model(model_path)
        self.sequence_length = 16  # Model expects sequences of 16 frames
        self.frame_queue = deque(maxlen=self.sequence_length)  # Queue to hold the sequence of frames

    def detect_violence(self, frame):
        # Preprocess the frame for the model and add it to the queue
        processed_frame = self.preprocess_frame(frame)
        self.frame_queue.append(processed_frame)

        # Only make predictions if we have a full sequence of frames
        if len(self.frame_queue) == self.sequence_length:
            # Stack the frames along the time axis to create a 5D tensor for prediction
            input_sequence = np.array(self.frame_queue)
            input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension

            # Make predictions using the violence detection model
            violence_pred = self.violence_model.predict(input_sequence)
            violence_label = "Non-Violence" if violence_pred[0][0] <= 0.5 else "Violence"
            return violence_label
        else:
            return "Processing..."  # Not enough frames yet for prediction

    def preprocess_frame(self, frame):
        # Convert frame to RGB and resize to 160x160 (expected input size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (160, 160))  # Resize to match model's expected input
        frame_array = img_to_array(frame_resized)
        frame_array /= 255.0  # Normalize pixel values

        return frame_array
