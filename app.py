import streamlit as st
import cv2
import os
import pandas as pd
from datetime import datetime
import time

# Set page configs
st.set_page_config(
    page_title="Real-time Face Detection",
    page_icon="./assets/faceman_cropped.png",
    layout="centered"
)

# Header Section
st.markdown(
    '<p style="text-align: center; font-size: 40px; font-weight: 550;">Realtime Frontal Face Detection</p>',
    unsafe_allow_html=True,
)
st.warning("NOTE: Click the arrow icon at Top-Left to open the Sidebar menu.")

# Sidebar Section
with st.sidebar:
    st.image("./assets/faceman_cropped.png", width=260)
    st.markdown(
        '<p style="font-size: 25px; font-weight: 550;">Face Detection Settings</p>',
        unsafe_allow_html=True,
    )

    # Sidebar options for face detection modes
    detection_mode = st.radio(
        "Choose Face Detection Mode",
        ('Home', 'Webcam Image Capture', 'Webcam Realtime Attendance Fill', 'Train Faces', 'Manual Attendance'),
        index=0
    )

# Main Page Content Based on Mode Selection
if detection_mode == "Home":
    st.title("Home 🏡")
    st.write("Welcome to the Real-Time Face Detection App!")

elif detection_mode == "Webcam Image Capture":
    st.header("Webcam Image Capture")
    run = st.checkbox("Run Webcam")
    FRAME_WINDOW = st.image([])  # Placeholder for video feed
    camera = cv2.VideoCapture(0)  # Open webcam

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    Enrollment = st.text_input("Enter Enrollment ID", value="333")
    Name = st.text_input("Enter Name", value="vivek")
    sampleNum = 0

    # Initialize DataFrame for attendance records
    csv_file = "attendance.csv"
    columns = ["Enrollment", "Name", "Date", "Time"]
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

    while run:
        ret, img = camera.read()
        if not ret:
            st.error("Failed to access the webcam. Please ensure it's connected.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save face images
            sampleNum += 1
            face_img = gray[y:y + h, x:x + w]
            os.makedirs("TrainingImage", exist_ok=True)
            cv2.imwrite(
                f"TrainingImage/{Name}.{Enrollment}.{sampleNum}.jpg", face_img
            )
            st.text(f"Image {sampleNum} saved for Enrollment ID: {Enrollment}")

            if sampleNum >= 20:
                st.success("Captured 20 images successfully.")
                run = False

                # Save attendance record to CSV
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")

                # Append data to the CSV
                df = pd.DataFrame([[Enrollment, Name, date, time_str]], columns=columns)
                df.to_csv(csv_file, mode='a', index=False, header=False)

                # Display CSV
                st.write("### Attendance Record")
                st.dataframe(pd.read_csv(csv_file))
                break

            # Add delay of 1 second between captures
            time.sleep(1)

        # Convert to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img_rgb)

    # Release camera resources
    camera.release()
    cv2.destroyAllWindows()

elif detection_mode == "Webcam Realtime Attendance Fill":
    st.header("Webcam Realtime Attendance Fill")
    st.write("This feature is under development.")

elif detection_mode == "Train Faces":
    st.header("Train Faces")
    st.write("This feature is under development.")

elif detection_mode == "Manual Attendance":
    st.header("Manual Attendance")
    st.write("This feature is under development.")
