import cv2
import numpy as np

# Load the pre-trained model and configuration file
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt',
    'MobileNetSSD_deploy.caffemodel'
)

# Define the classes for the model (including "person")
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
           'bottle', 'bus', 'car', 'cat', 'chair', 
           'cow', 'dining table', 'dog', 'horse', 
           'motorbike', 'person', 'potted plant', 
           'sheep', 'sofa', 'train', 'tv/monitor']

def count_people_in_frame():
    # Set up webcam feed
    cap = cv2.VideoCapture(0)  # Use webcam
    
    if not cap.isOpened():
        print("Could not open webcam.")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Prepare the frame for the model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        
        # Forward pass to get detections
        detections = net.forward()

        person_count = 0

        # Count people in frame
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # Confidence threshold
                class_id = int(detections[0, 0, i, 1])
                if classes[class_id] == 'person':
                    person_count += 1
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Display the count on the frame
        cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Show the output frame
        cv2.imshow('Frame', frame)
        
        # Return the count so it can be used by other parts of the project
        print(f"People Count: {person_count}")
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    return person_count  # Last counted value when exiting

# Call the function (for demonstration)
total_count = count_people_in_frame()
print(f"Final People Count: {total_count}")
