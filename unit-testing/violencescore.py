import cv2
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from torchvision import transforms

# Load the image classification model and feature extractor for violence detection
model = ViTForImageClassification.from_pretrained("jaranohaal/vit-base-violence-detection")
feature_extractor = ViTFeatureExtractor.from_pretrained("jaranohaal/vit-base-violence-detection")

# Define a transformation for converting OpenCV frames to PIL images
transform = transforms.ToPILImage()

def analyze_frame(frame):
    # Convert the OpenCV frame to a PIL image
    pil_image = transform(frame)

    # Prepare inputs for the classification model
    inputs = feature_extractor(images=pil_image, return_tensors="pt")

    # Forward pass through the model to get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Get the predicted label from the model's config
    predicted_label = model.config.id2label[predicted_class_idx]

    # Return True if any harmful action is detected, otherwise False
    return "violence" in predicted_label.lower(), predicted_label

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    yes_count = 0
    no_count = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame.")
            break
        
        # Analyze each frame for violence detection
        violence_detected, prediction = analyze_frame(frame)
        total_frames += 1
        
        # Update counts based on the prediction
        if violence_detected:
            yes_count += 1
        else:
            no_count += 1
        
        # Print result for the individual frame
        print(f"Frame {total_frames}: Prediction - {prediction}")

        # Calculate and print running average scores
        yes_avg = yes_count / total_frames * 100
        no_avg = no_count / total_frames * 100
        print(f"Running Average - Violence Detected: {yes_avg:.2f}% | No Violence: {no_avg:.2f}%")

        # Display the current frame
        cv2.imshow("Violence Detection", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final summary for all frames
    print("\nFinal Summary:")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Violence Detected in {yes_count} frames ({yes_avg:.2f}%)")
    print(f"No Violence in {no_count} frames ({no_avg:.2f}%)")

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Path to the input video file
video_path = "C:/Users/Manamnath tiwari/Downloads/Pennsylvania baseball player cut over domestic abuse video.mp4"
process_video(video_path)
