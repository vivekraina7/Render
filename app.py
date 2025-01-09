# import streamlit as st
# import os
# import pandas as pd
# from datetime import datetime
# from PIL import Image
# import cv2
# import numpy as np

# # Set page configs
# st.set_page_config(
#     page_title="Real-time Face Detection",
#     page_icon="./assets/faceman_cropped.png",
#     layout="centered"
# )

# # Header Section
# st.markdown(
#     '<p style="text-align: center; font-size: 40px; font-weight: 550;">Realtime Frontal Face Detection</p>',
#     unsafe_allow_html=True,
# )
# st.warning("NOTE: Click the arrow icon at Top-Left to open the Sidebar menu.")

# # Sidebar Section
# with st.sidebar:
#     st.image("./assets/faceman_cropped.png", width=260)
#     st.markdown(
#         '<p style="font-size: 25px; font-weight: 550;">Face Detection Settings</p>',
#         unsafe_allow_html=True,
#     )

#     # Sidebar options for face detection modes
#     detection_mode = st.radio(
#         "Choose Face Detection Mode",
#         ('Home', 'File Upload for Image Capture', 'Manual Attendance','Train Faces','Recognize Faces','Recognize from Camera'),
#         index=0
#     )

# # Load Haar Cascade for face detection
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Main Page Content Based on Mode Selection
# if detection_mode == "Home":
#     st.title("Home 🏡")
#     st.write("Welcome to the Real-Time Face Detection App!")

# elif detection_mode == "File Upload for Image Capture":
#     st.header("File Upload for Image Capture")
#     Enrollment = st.text_input("Enter Enrollment ID", value="333")
#     Name = st.text_input("Enter Name", value="vivek")

#     if Enrollment and Name:
#         # Create a folder for the student
#         folder_path = os.path.join("TrainingImages", f"{Name}_{Enrollment}")
#         os.makedirs(folder_path, exist_ok=True)

#         st.write(f"Upload at least 5 images for {Name} ({Enrollment}).")

#         # File uploader to accept multiple images
#         uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

#         if uploaded_files:
#             saved_files = []

#             for i, uploaded_file in enumerate(uploaded_files):
#                 if i >= 5:  # Limit to 5 images
#                     break

#                 # Open the uploaded image
#                 image = Image.open(uploaded_file)
                
#                 # Ensure the image is in RGB format
#                 image = image.convert("RGB")
#                 image_np = np.array(image)

#                 # Convert image to grayscale for face detection
#                 gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

#                 # Detect faces
#                 faces = detector.detectMultiScale(gray, 1.3, 5)

#                 for (x, y, w, h) in faces:
#                     # Crop the face region
#                     face = image_np[y:y + h, x:x + w]
#                     face_image = Image.fromarray(face)

#                     # Save the cropped face image
#                     file_path = os.path.join(folder_path, f"face_{i+1}.jpg")
#                     face_image.save(file_path)
#                     saved_files.append(file_path)

#             st.success(f"Successfully saved {len(saved_files)} face images in folder: {folder_path}")

#             # Save details in a CSV
#             csv_file = "attendance.csv"
#             columns = ["Enrollment", "Name", "Date", "Time"]
#             now = datetime.now()
#             date = now.strftime("%Y-%m-%d")
#             time_str = now.strftime("%H:%M:%S")

#             if not os.path.exists(csv_file):
#                 pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

#             df = pd.DataFrame([[Enrollment, Name, date, time_str]], columns=columns)
#             df.to_csv(csv_file, mode='a', index=False, header=False)

#             st.write("### Attendance Record")
#             st.dataframe(pd.read_csv(csv_file))



# elif detection_mode == "Train Faces":
#     st.header("Train Faces")

#     # Path for face image database
#     path = 'TrainingImages'

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     # Function to get the images and label data
#     def getImagesAndLabels(path):
#         faceSamples = []
#         ids = []

#         for root, _, files in os.walk(path):
#             for file in files:
#                 try:
#                     imagePath = os.path.join(root, file)
#                     PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
#                     img_numpy = np.array(PIL_img, 'uint8')

#                     id = int(os.path.basename(root).split("_")[-1])  # Extract ID from folder name
#                     faces = detector.detectMultiScale(img_numpy)

#                     for (x, y, w, h) in faces:
#                         faceSamples.append(img_numpy[y:y + h, x:x + w])
#                         ids.append(id)
#                 except Exception as e:
#                     st.warning(f"Skipped file {file}: {e}")

#         return faceSamples, ids

#     st.text("\n [INFO] Training faces. It will take a few seconds. Wait ...")
#     faces, ids = getImagesAndLabels(path)
#     recognizer.train(faces, np.array(ids))

#     # Save the model into trainer/trainer.yml
#     os.makedirs("TrainingImageLabel", exist_ok=True)
#     recognizer.write('TrainingImageLabel/Trainer.yml')

#     # Print the number of faces trained and end program
#     st.text(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")



# elif detection_mode == "Recognize Faces":
#     st.header("Recognize Faces")

#     # File uploader to input an image
#     uploaded_file = st.file_uploader("Upload an Image to Recognize Faces", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         image_np = np.array(image)

#         # Convert image to grayscale for face detection
#         gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

#         # Detect faces
#         faces = detector.detectMultiScale(gray, 1.3, 5)

#         # Load the trained recognizer model
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         recognizer.read('TrainingImageLabel/Trainer.yml')

#         recognized_faces = []

#         for (x, y, w, h) in faces:
#             face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#             recognized_faces.append((face_id, confidence))

#             # Draw a rectangle around the face
#             cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(
#                 image_np,
#                 f"ID: {face_id}, Conf: {round(100 - confidence, 2)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 0, 0),
#                 2
#             )

#         # Convert back to PIL for display
#         result_image = Image.fromarray(image_np)
#         st.image(result_image, caption="Recognized Faces", use_column_width=True)

#         if recognized_faces:
#             st.write("### Recognized Faces")
#             for face_id, confidence in recognized_faces:
#                 st.write(f"Face ID: {face_id}, Confidence: {round(100 - confidence, 2)}%")
#         else:
#             st.write("No faces recognized.")

# elif detection_mode == "Recognize from Camera":
#     st.header("Recognize Faces from Camera")

#     enable_camera = st.checkbox("Enable Camera")
#     picture = st.camera_input("Take a picture", disabled=not enable_camera)

#     if picture:
#         image = Image.open(picture)
#         image_np = np.array(image)

#         # Convert image to grayscale for face detection
#         gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

#         # Detect faces
#         faces = detector.detectMultiScale(gray, 1.3, 5)

#         # Load the trained recognizer model
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         recognizer.read('TrainingImageLabel/Trainer.yml')

#         recognized_faces = []

#         for (x, y, w, h) in faces:
#             face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#             recognized_faces.append((face_id, confidence))

#             # Draw a rectangle around the face
#             cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(
#                 image_np,
#                 f"ID: {face_id}, Conf: {round(100 - confidence, 2)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 0, 0),
#                 2
#             )

#         # Convert back to PIL for display
#         result_image = Image.fromarray(image_np)
#         st.image(result_image, caption="Recognized Faces", use_column_width=True)

#         if recognized_faces:
#             st.write("### Recognized Faces")
#             for face_id, confidence in recognized_faces:
#                 st.write(f"Face ID: {face_id}, Confidence: {round(100 - confidence, 2)}%")
#         else:
#             st.write("No faces recognized.")
            
# elif detection_mode == "Manual Attendance":
#     st.header("Manual Attendance")
#     st.write("This feature is under development.")

import streamlit as st
import os
import pandas as pd
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        ('Home', 'File Upload for Image Capture', 'Train Faces', 'Recognize Faces', 'Recognize from Camera','Attendance Analysis'),
        index=0
    )

# Load Haar Cascade for face detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to save recognized faces to a CSV file
def save_recognition_to_csv(subject, recognized_faces):
    os.makedirs("RecognizedFaces", exist_ok=True)  # Ensure the directory exists
    date = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join("RecognizedFaces", f"{subject}_{date}.csv")
    columns = ["Face ID", "Confidence", "Date", "Time"]

    if not os.path.exists(csv_file):
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    
    rows = [[face_id, round(100 - confidence, 2), date, time_str] for face_id, confidence in recognized_faces]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(csv_file, mode='a', index=False, header=False)

# Function for Attendance Analysis
def attendance_analysis():
    st.header("Attendance Analysis")
    folder_path = "RecognizedFaces"
    attendance_csv = "attendance.csv"

    if not os.path.exists(folder_path):
        st.error("No attendance data available.")
        return

    if not os.path.exists(attendance_csv):
        st.error("No master attendance file (attendance.csv) found.")
        return

    all_students = pd.read_csv(attendance_csv)
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not all_files:
        st.error("No attendance CSV files found in the folder.")
        return

    attendance_data = {}
    for file in all_files:
        subject = file.split("_")[0]
        csv_path = os.path.join(folder_path, file)
        df = pd.read_csv(csv_path)
        if subject not in attendance_data:
            attendance_data[subject] = []
        attendance_data[subject].extend(df["Face ID"].tolist())

    # Calculate attendance percentage for each student
    student_attendance = {}
    for subject, face_ids in attendance_data.items():
        subject_students = all_students
        enrolled_students = set(subject_students['Enrollment'].tolist())
        present_students = set(face_ids)
        absent_students = enrolled_students - present_students

        for student in enrolled_students:
            if student not in student_attendance:
                student_attendance[student] = {}
            present_count = face_ids.count(student)
            total_classes = len(set(face_ids))
            student_attendance[student][subject] = {
                "present": present_count,
                "total": total_classes,
                "percentage": (present_count / total_classes) * 100 if total_classes > 0 else 0
            }

        # Display subject-wise data
        st.subheader(f"Subject: {subject}")
        st.write(f"- Total Classes: {len(set(face_ids))}")
        st.write(f"- Students Present: {len(present_students)}")
        st.write(f"- Students Absent: {len(absent_students)}")
        st.write(f"- Present Student IDs: {', '.join(map(str, present_students))}")
        st.write(f"- Absent Student IDs: {', '.join(map(str, absent_students))}")

        # Bar Chart
        fig, ax = plt.subplots()
        ax.bar(["Present", "Absent"], [len(present_students), len(absent_students)])
        ax.set_title(f"Attendance for {subject}")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Individual attendance percentages
    st.subheader("Individual Student Attendance Percentage")
    for student, subjects in student_attendance.items():
        st.write(f"### Student ID: {student}")
        for subject, stats in subjects.items():
            st.write(f"- {subject}: {stats['present']} days present out of {stats['total']} classes")
            st.progress(stats['percentage'] / 100)
# Main Page Content Based on Mode Selection
if detection_mode == "Home":
    st.title("Home 🏡")
    st.write("Welcome to the Real-Time Face Detection App!")

elif detection_mode == "Attendance Analysis":
    attendance_analysis()

elif detection_mode == "File Upload for Image Capture":
    st.header("File Upload for Image Capture")
    Enrollment = st.text_input("Enter Enrollment ID", value="333")
    Name = st.text_input("Enter Name", value="vivek")

    if Enrollment and Name:
        # Create a folder for the student
        folder_path = os.path.join("TrainingImages", f"{Name}_{Enrollment}")
        os.makedirs(folder_path, exist_ok=True)

        st.write(f"Upload at least 5 images for {Name} ({Enrollment}).")

        # File uploader to accept multiple images
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            saved_files = []

            for i, uploaded_file in enumerate(uploaded_files):
                if i >= 5:  # Limit to 5 images
                    break

                # Open the uploaded image
                image = Image.open(uploaded_file)
                
                # Ensure the image is in RGB format
                image = image.convert("RGB")
                image_np = np.array(image)

                # Convert image to grayscale for face detection
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

                # Detect faces
                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    # Crop the face region
                    face = image_np[y:y + h, x:x + w]
                    face_image = Image.fromarray(face)

                    # Save the cropped face image
                    file_path = os.path.join(folder_path, f"face_{i+1}.jpg")
                    face_image.save(file_path)
                    saved_files.append(file_path)

            st.success(f"Successfully saved {len(saved_files)} face images in folder: {folder_path}")

            # Save details in a CSV
            csv_file = "attendance.csv"
            columns = ["Enrollment", "Name", "Date", "Time"]
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            if not os.path.exists(csv_file):
                pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

            df = pd.DataFrame([[Enrollment, Name, date, time_str]], columns=columns)
            df.to_csv(csv_file, mode='a', index=False, header=False)

            st.write("### Attendance Record")
            st.dataframe(pd.read_csv(csv_file))



elif detection_mode == "Train Faces":
    st.header("Train Faces")

    # Path for face image database
    path = 'TrainingImages'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Function to get the images and label data
    def getImagesAndLabels(path):
        faceSamples = []
        ids = []

        for root, _, files in os.walk(path):
            for file in files:
                try:
                    imagePath = os.path.join(root, file)
                    PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
                    img_numpy = np.array(PIL_img, 'uint8')

                    id = int(os.path.basename(root).split("_")[-1])  # Extract ID from folder name
                    faces = detector.detectMultiScale(img_numpy)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y + h, x:x + w])
                        ids.append(id)
                except Exception as e:
                    st.warning(f"Skipped file {file}: {e}")

        return faceSamples, ids

    st.text("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    os.makedirs("TrainingImageLabel", exist_ok=True)
    recognizer.write('TrainingImageLabel/Trainer.yml')

    # Print the number of faces trained and end program
    st.text(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")
elif detection_mode == "Recognize Faces":
    st.header("Recognize Faces")

    subject = st.text_input("Enter Subject", value="Python")

    # File uploader to input an image
    uploaded_file = st.file_uploader("Upload an Image to Recognize Faces", type=["jpg", "jpeg", "png"])

    if uploaded_file and subject:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(gray, 1.3, 5)

        # Load the trained recognizer model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('TrainingImageLabel/Trainer.yml')

        recognized_faces = []

        for (x, y, w, h) in faces:
            face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            recognized_faces.append((face_id, confidence))

            # Draw a rectangle around the face
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image_np,
                f"ID: {face_id}, Conf: {round(100 - confidence, 2)}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        # Convert back to PIL for display
        result_image = Image.fromarray(image_np)
        st.image(result_image, caption="Recognized Faces", use_column_width=True)

        if recognized_faces:
            st.write("### Recognized Faces")
            for face_id, confidence in recognized_faces:
                st.write(f"Face ID: {face_id}, Confidence: {round(100 - confidence, 2)}%")

            # Save recognized faces to CSV
            save_recognition_to_csv(subject, recognized_faces)

            st.success(f"Data saved to CSV file for subject: {subject}")
        else:
            st.write("No faces recognized.")

elif detection_mode == "Recognize from Camera":
    st.header("Recognize Faces from Camera")

    subject = st.text_input("Enter Subject", value="Python")
    enable_camera = st.checkbox("Enable Camera")
    picture = st.camera_input("Take a picture", disabled=not enable_camera)

    if picture and subject:
        image = Image.open(picture)
        image_np = np.array(image)

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(gray, 1.3, 5)

        # Load the trained recognizer model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('TrainingImageLabel/Trainer.yml')

        recognized_faces = []

        for (x, y, w, h) in faces:
            face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            recognized_faces.append((face_id, confidence))

            # Draw a rectangle around the face
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image_np,
                f"ID: {face_id}, Conf: {round(100 - confidence, 2)}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        # Convert back to PIL for display
        result_image = Image.fromarray(image_np)
        st.image(result_image, caption="Recognized Faces", use_column_width=True)

        if recognized_faces:
            st.write("### Recognized Faces")
            for face_id, confidence in recognized_faces:
                st.write(f"Face ID: {face_id}, Confidence: {round(100 - confidence, 2)}%")

            # Save recognized faces to CSV
            save_recognition_to_csv(subject, recognized_faces)

            st.success(f"Data saved to CSV file for subject: {subject}")
        else:
            st.write("No faces recognized.")
