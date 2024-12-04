import os  
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque

class ViolenceDetector:
    def __init__(self):
        model_path = os.path.join(r"C:\Users\Manamnath tiwari\OneDrive\Desktop\Mirage-Real Time Violence Detection", "violence_detection_model12.keras")
        
        self.violence_model = load_model(model_path)
        self.sequence_length = 16  
        self.frame_queue = deque(maxlen=self.sequence_length) 

    def detect_violence(self, frame):
        processed_frame = self.preprocess_frame(frame)
        self.frame_queue.append(processed_frame)


        if len(self.frame_queue) == self.sequence_length:
            input_sequence = np.array(self.frame_queue)
            input_sequence = np.expand_dims(input_sequence, axis=0)  

            violence_pred = self.violence_model.predict(input_sequence)
            violence_label = "Non-Violence" if violence_pred[0][0] <= 0.5 else "Violence"
            return violence_label
        else:
            return "Processing..." 

    def preprocess_frame(self, frame):
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (160, 160)) 
        frame_array = img_to_array(frame_resized)
        frame_array /= 255.0  

        return frame_array
