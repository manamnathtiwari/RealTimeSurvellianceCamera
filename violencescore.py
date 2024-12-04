import cv2
import torch
import time
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from torchvision import transforms

# Load the model and feature extractor for violence detection
model = ViTForImageClassification.from_pretrained("jaranohaal/vit-base-violence-detection")
feature_extractor = ViTFeatureExtractor.from_pretrained("jaranohaal/vit-base-violence-detection")

# Transform for converting OpenCV frames to PIL images
transform = transforms.ToPILImage()

def analyze_frame(frame):
    """Analyzes a single frame for violence."""
    pil_image = transform(frame)
    inputs = feature_extractor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Interpret the label
    predicted_label = model.config.id2label[predicted_class_idx]
    return "violence" in predicted_label.lower(), predicted_label

def process_webcam_feed():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    violence_count = 0  # Counter for violence detections in the past 5 minutes
    total_frames = 0
    start_time = time.time()
    interval_count = 0  # To manage 3-second intervals

    emergency_trigger_count = 0  # Track if continuous violence over 5 minutes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        # Analyze the frame every 3 seconds
        current_time = time.time()
        if current_time - start_time >= 3:  # Every 3 seconds
            start_time = current_time
            interval_count += 1

            # Analyze the frame for violence detection
            violence_detected, prediction = analyze_frame(frame)
            if violence_detected:
                violence_count += 1  # Count frames where violence is detected
                print("Violence Detected")
            else:
                print("No Violence Detected")
            
            # Calculate average for this interval
            avg_violence_detected = 1 if violence_detected else 0
            print(f"Interval {interval_count}: Violence Detected - {avg_violence_detected}")

            # Check if continuous violence has been detected over 5 minutes (100 intervals)
            if interval_count >= 100:
                if violence_count >= 100:  # All intervals reported violence
                    emergency_trigger_count = 2
                    print("Emergency Triggered: Continuous Violence Detected for 5 Minutes!")
                else:
                    emergency_trigger_count = 1 if violence_count > 0 else 0
                interval_count = 0  # Reset interval count for next 5-minute check
                violence_count = 0  # Reset the violence count
            
            # Display current interval detection
            if emergency_trigger_count == 2:
                cv2.putText(frame, "EMERGENCY: Continuous Violence Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif avg_violence_detected == 1:
                cv2.putText(frame, "Violence Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No Violence", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print(f"Average Violence Detection over last 3 seconds: {avg_violence_detected}")
            print(f"Emergency Status: {emergency_trigger_count}")
        
        # Display the current frame
        cv2.imshow("Violence Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time webcam violence detection
process_webcam_feed()
