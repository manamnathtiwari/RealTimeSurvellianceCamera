import streamlit as st
import cv2
import time
import mediapipe as mp
from mtcnn import MTCNN
import numpy as np
from PIL import Image
import tempfile
import os


from Person_Detection import detect_person
from gender_Detection import classify_gender
from emotion_Detection import EmotionDetector
from Centroid_Tracker import CentroidTracker
from SOS_Condition import is_female_surrounded
from pose import detect_action
from Telebot_alert import send_telegram_alert
from facial_expression import classify_face, draw_selected_landmarks
from Violence import ViolenceDetector

def main():
    
    st.set_page_config(page_title="Surveillance System", layout="wide")
    
    
    st.title("Real-time Harassment Surveillance System")
    st.markdown("""
    This system monitors for:
    - Person detection and tracking
    - Gender classification
    - Emotion detection
    - Violence detection
    - Suspicious behavior patterns
    """)

    
    st.sidebar.header("Controls")
    
    
    source_option = st.sidebar.radio(
        "Select Video Source",
        ["Webcam", "Upload Video"]
    )

    # Alert settings
    st.sidebar.header("Alert Settings")
    enable_night_alerts = st.sidebar.checkbox("Enable Night Time Alerts", value=True)
    enable_surrounding_alerts = st.sidebar.checkbox("Enable Surrounding Alerts", value=True)
    enable_violence_alerts = st.sidebar.checkbox("Enable Violence Alerts", value=True)

    
    start_analysis = st.sidebar.button('Start Analysis', key="start_button")
    stop_analysis = st.sidebar.button('Stop Analysis', key="stop_button")

    
    tracker = CentroidTracker()
    detector = MTCNN()
    emotion_detector = EmotionDetector()
    violence_detector = ViolenceDetector()
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Video display
        video_placeholder = st.empty()

    with col2:
        # Stats and alerts
        st.subheader("Real-time Statistics")
        stats_placeholder = st.empty()

        st.subheader("Active Alerts")
        alert_placeholder = st.empty()

    # Video Capture Setup
    cap = None
    if start_analysis:
        if source_option == "Webcam":
            cap = cv2.VideoCapture(0)
        elif source_option == "Upload Video":
            uploaded_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                tfile.close()
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.warning("Please upload a video file")
                return

    # Processing loop
    active_alerts = []
    while start_analysis and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Person detection
        person_boxes = detect_person(frame)
        n = len(person_boxes)

        # Reset counts
        male_count = 0
        female_count = 0
        mbbox = []

        # Update tracker
        objects = tracker.update(person_boxes)

        # Process each detected person
        for i, (objectID, centroid) in enumerate(objects.items()):
            if objectID < len(person_boxes):
                x1, y1, x2, y2 = map(int, person_boxes[i])
                person_img = frame[y1:y2, x1:x2]

                # Face detection
                faces = detector.detect_faces(person_img)
                if faces:
                    face = faces[0]
                    x, y, width, height = face['box']
                    face_img = person_img[y:y+height, x:x+width]

                    # Process face
                    results = mp_holistic.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    if results.face_landmarks:
                        # Get classifications
                        face_class = classify_face(results.face_landmarks)
                        gender_label = classify_gender(face_img)
                        emotion_label = emotion_detector.detect_emotions(face_img)
                        pose_action = detect_action(results.pose_landmarks)
                        violence_label = violence_detector.detect_violence(person_img)

                        # Update counts and store data
                        if gender_label:
                            if 'male' in gender_label:
                                male_count += 1
                                mbbox.append(person_img)
                            elif 'female' in gender_label:
                                female_count += 1
                                female_bbox = person_img

                        # Generate alerts
                        current_alerts = []

                        if enable_night_alerts:
                            current_hour = time.localtime().tm_hour
                            if n == 1 and 'female' in gender_label and (current_hour >= 18 or current_hour < 6):
                                current_alerts.append("⚠️ Female detected alone at night")

                        if enable_surrounding_alerts:
                            if female_count == 1 and n > 2 and (face_class == 'Fear' or face_class == 'Distress'):
                                if is_female_surrounded(female_bbox, mbbox):
                                    current_alerts.append("🚨 Female surrounded by multiple males")

                        if enable_violence_alerts and 'Violence' in violence_label:
                            current_alerts.append("⚠️ Potential violence detected")

                        # Draw annotations
                        label = f'ID {objectID}: {gender_label}, {face_class}, {emotion_label}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, tuple(map(int, centroid)), 4, (255, 0, 0), -1)

                        # Update alerts
                        active_alerts = current_alerts

        # Convert frame for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame)

        # Update statistics
        stats_placeholder.markdown(f"""
        - Total Persons: {n}
        - Males: {male_count}
        - Females: {female_count}
        """)

        # Update alerts
        if active_alerts:
            alert_text = "\n".join(active_alerts)
            alert_placeholder.error(alert_text)
        else:
            alert_placeholder.success("No active alerts")

        # Check if 'Stop Analysis' button was pressed
        if stop_analysis:
            break

    # Cleanup
    if cap is not None:
        cap.release()
        if source_option == "Upload Video":
            os.unlink(tfile.name)

if __name__ == "__main__":
    main()
