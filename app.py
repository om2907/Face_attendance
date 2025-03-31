import streamlit as st
import cv2
import os
import numpy as np
import pickle
import face_recognition
from datetime import datetime
from datetime import date
import math
import time
import csv
import os
import datetime
import pandas as pd

DATA_PATH = "Data"

# Load existing face data
def load_data():
    if os.path.exists(f"{DATA_PATH}/encodings.pkl"):
        with open(f"{DATA_PATH}/encodings.pkl", "rb") as f:
            encodings = pickle.load(f)
    else:
        encodings = []

    if os.path.exists(f"{DATA_PATH}/names.pkl"):
        with open(f"{DATA_PATH}/names.pkl", "rb") as f:
            names = pickle.load(f)
    else:
        names = []

    return names, encodings

# Save updated face data
def save_data(names, encodings):
    with open(f"{DATA_PATH}/names.pkl", "wb") as f:
        pickle.dump(names, f)
    with open(f"{DATA_PATH}/encodings.pkl", "wb") as f:
        pickle.dump(encodings, f)

# Video Face Registration Function
def register_face():
    st.subheader("Face Registration")
    name = st.text_input("Enter Your Name")
    upload_option = st.radio("Choose Registration Method", ("Live Camera", "Upload Image"))

    if not name.strip():
        st.warning("⚠️ Name cannot be empty!")
        return

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        if uploaded_file :
            file_path = "input_data/registered_img.jpg"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Display uploaded image on the right
            col1, col2 = st.columns([3, 1])
            with col2:
                st.image(uploaded_file, caption="Uploaded Image", width=150)

            st.info("Processing image...")
            progress_bar = st.progress(0)

            # Simulating processing delay
            for percent_complete in range(1, 101):
                time.sleep(0.01)
                progress_bar.progress(percent_complete)

            # Load and process image
            image = face_recognition.load_image_file(file_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 0:
                st.error("❌ No face detected in uploaded image.")
            elif len(face_encodings) > 1:
                st.warning("⚠️ Multiple faces detected! Please upload an image with only one face.")
            else:
                names, encodings = load_data()
                names.append(name)
                encodings.append(face_encodings[0])
                save_data(names, encodings)
                st.success("✅ Face Registered Successfully!")

    elif upload_option == "Live Camera":
        if st.button("Start Registration"):
            video = cv2.VideoCapture(0)
            faces_encodings = []

            st.info("Align your face properly in the camera...")
            progress_bar = st.progress(0)

            frame_count = 0
            while len(faces_encodings) < 10:
                ret, frame = video.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                if face_encodings:
                    faces_encodings.append(face_encodings[0])  # Take first detected face

                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                cv2.imshow("Registration", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

                frame_count += 1
                progress_bar.progress(min(len(faces_encodings) * 10, 100))  # Update progress bar

            video.release()
            cv2.destroyAllWindows()
            progress_bar.empty()

            if len(faces_encodings) == 10:
                names, encodings = load_data()
                names += [name] * 10
                encodings += faces_encodings
                save_data(names, encodings)
                st.success("✅ Face Registered Successfully from Camera!")
            else:
                st.error("❌ Face registration was not successful.")

#  Face Recognition
MAX_DURATION = 7  # Set max video duration (seconds)
MAX_FRAMES_ANALYZED = 100  # Limit frames analyzed
def recognize_faces_vid():
    st.subheader("Upload Video for Face Recognition")
    keyword_file()
    uploaded_file = st.file_uploader("Upload a video (Max: 7 sec)", type=["mp4", "avi", "mov"])

    if uploaded_file:
        file_path = "uploaded_video.mp4"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Check video duration
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
        duration = total_frames / fps  # Calculate duration in seconds

        if duration > MAX_DURATION:
            st.error(f"Video too long! Maximum allowed duration is {MAX_DURATION} seconds.")
            return  # Stop processing

        st.success(f"Video uploaded successfully! Duration: {duration:.2f} sec")

        # Load registered faces
        names, encodings = load_data()
        if len(names) == 0 or len(encodings) == 0:
            st.error("No registered faces found.")
            return

        recognized_faces = set()
        frames_analyzed = 0
        frame_count = 0

        # Progress bar
        progress_bar = st.progress(0)
        total_frames_to_analyze = min(MAX_FRAMES_ANALYZED, total_frames // 20)  # Since we analyze every 20th frame

        while frames_analyzed < MAX_FRAMES_ANALYZED:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1  # Track total frames

            if frame_count % 20 != 0:  # Process every 20th frame
                continue

            frames_analyzed += 1  # Count analyzed frames

            # Update progress bar
            progress_bar.progress(frames_analyzed / total_frames_to_analyze)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)
                if True in matches:
                    matched_idx = [i for (i, b) in enumerate(matches) if b]
                    name = names[max(set(matched_idx), key=matched_idx.count)]
                    recognized_faces.add(name)

        progress_bar.empty()  # Remove progress bar after completion
        st.write("Recognized Faces:", ", ".join(recognized_faces) if recognized_faces else "No faces recognized.")
        st.write(f"Total frames analyzed: {frames_analyzed}")

        video.release()
        save_attendance(recognized_faces)
        st.success("Attendance recorded successfully!")
        return recognized_faces


# Image Face Recognition
def recognize_faces_img():
    st.subheader("Upload Image for Face Recognition")
    keyword_file()
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_path = "uploaded_img.jpg"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
            col1, col2 = st.columns([3, 1])  # 3:1 ratio to keep the image on the right in a smaller area
            with col2:
                st.image(uploaded_file, caption="Uploaded image", width=150)  # Adjust width as needed

        st.success("Image uploaded successfully!")
        
        # Load registered faces
        names, encodings = load_data()
        if len(names) == 0 or len(encodings) == 0:
            st.error("No registered faces found.")
            return

        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        recognized_faces = set()
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_faces = len(face_encodings)

        for idx, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)
            if True in matches:
                matched_idx = [i for (i, b) in enumerate(matches) if b]
                name = names[max(set(matched_idx), key=matched_idx.count)]
                recognized_faces.add(name)

            progress_bar.progress((idx + 1) / total_faces if total_faces else 1.0)
            progress_text.text(f"Processing face {idx + 1}/{total_faces}...")
            time.sleep(0.1)

        progress_bar.empty()
        progress_text.text("Face recognition completed!")

        st.write("Recognized Faces:", ", ".join(recognized_faces))
        save_attendance(recognized_faces)  # Save recognized faces to CSV
        return recognized_faces
        return keyword


# Live Face Recognition Function
def live_face_recognition():
    st.subheader("Live Face Recognition")
    names, encodings = load_data()

    if len(names) == 0 or len(encodings) == 0:
        st.error("No registered faces available for recognition.")
        return

    video = cv2.VideoCapture(0)
    recognized_names = set()  # Store unique recognized names

    stframe = st.empty()  # Streamlit frame placeholder
    names_placeholder = st.empty()  # Placeholder for recognized names

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_names.clear()  # Reset recognized names on each frame

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)
            if True in matches:
                matched_idx = [i for (i, b) in enumerate(matches) if b]
                name = names[max(set(matched_idx), key=matched_idx.count)]
                recognized_names.add(name)

        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")
        names_placeholder.write(f"**Recognized Names:** {', '.join(recognized_names)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Face Deletion Function
def delete_face():
    st.subheader("Delete Registered Faces")

    # Load registered names and encodings
    names, encodings = load_data()

    if len(names) == 0:
        st.warning("No registered faces found.")
        return

    # Get unique names
    unique_names = sorted(set(names))

    # Initialize session state for checkboxes
    if "selected_faces" not in st.session_state:
        st.session_state.selected_faces = {name: False for name in unique_names}

    # Display checkboxes for each name
    st.write("### Select Faces to Delete:")
    for name in unique_names:
        st.session_state.selected_faces[name] = st.checkbox(name, st.session_state.selected_faces[name])

    # Buttons for deletion and clearing selections
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Delete Selected Faces", type="primary"):
            selected_names = [name for name, checked in st.session_state.selected_faces.items() if checked]
            if selected_names:
                new_names = []
                new_encodings = []

                for i in range(len(names)):
                    if names[i] not in selected_names:
                        new_names.append(names[i])
                        new_encodings.append(encodings[i])

                save_data(new_names, new_encodings)

                st.success(f"Deleted face data for: {', '.join(selected_names)}")
                st.session_state.selected_faces = {name: False for name in unique_names}  # Reset checkboxes
                st.rerun()
            else:
                st.warning("No faces selected for deletion.")

    with col2:
        if st.button("Clear All Selections"):
            st.session_state.selected_faces = {name: False for name in unique_names}
            st.rerun()


def keyword_file():
    global keyword
    full_keyword = st.text_input("Enter a keyword for the filename", value="", key="keyword")
    keyword = ("_"+full_keyword)
    return keyword




# Define the attendance folder path
ATTENDANCE_FOLDER = "attendance"

# Ensure the folder exists
os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

def get_attendance_filename():
    """Returns the attendance CSV filename for today's date."""
    today = date.today().strftime("%d-%m-%Y")  # Format: DD-MM-YYYY
    return os.path.join(ATTENDANCE_FOLDER, f"{today}{keyword}.csv")

def save_attendance(recognized_faces):
    """Saves recognized face names with timestamps in today's CSV file."""
    if not recognized_faces:
        return  # No names to save

    filename = get_attendance_filename()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if file exists, if not create with headers
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=["Name", "Timestamp"])
        df.to_csv(filename, index=False)

    # Load existing data
    df = pd.read_csv(filename)

    # Append new attendance records (avoid duplicates within the same day)
    new_records = pd.DataFrame([{"Name": name, "Timestamp": timestamp} for name in recognized_faces])

    df = pd.concat([df, new_records], ignore_index=True)

    # Save updated attendance data
    df.to_csv(filename,index=False)
    st.success("**Attendance saved successfully in " + filename + "**")


def display_attendance_records():
    """Displays attendance records and allows users to delete individual records or the entire file."""
    st.subheader("📋 Attendance Records")

    # Ensure the folder exists
    if not os.path.exists(ATTENDANCE_FOLDER):
        os.makedirs(ATTENDANCE_FOLDER)

    # Get all attendance CSV files
    files = sorted([f for f in os.listdir(ATTENDANCE_FOLDER) if f.endswith(".csv")], reverse=True)

    if not files:
        st.warning("No attendance records found.")
        return

    # Show available dates for selection
    selected_file = st.selectbox("Select a date to view attendance", files)

    if selected_file:
        filepath = os.path.join(ATTENDANCE_FOLDER, selected_file)
        df = pd.read_csv(filepath)

        if df.empty:
            st.info("No records found for this date.")
        else:
            st.write(f"### Attendance for {selected_file.replace('.csv', '')}")

            # Display the dataframe with a delete button for each row
            for index, row in df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{row['Name']}   |   {row['Timestamp']}")
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{index}_{selected_file}"):
                        df = df.drop(index)
                        df.to_csv(filepath, index=False)
                        st.success(f"Deleted record of {row['Name']}.")
                        st.rerun()

            # Download button
            csv_file = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv_file, selected_file, "text/csv")

        # Delete entire file button
        if st.button("🗑️ Delete Entire File", key=f"delete_file_{selected_file}"):
            os.remove(filepath)
            st.warning(f"Deleted {selected_file} successfully!")
            st.rerun()





    
        























# Streamlit UI
st.markdown("<h2 style='text-align: center;'>Face Recognition System</h2>", unsafe_allow_html=True)


menu = st.sidebar.radio("Menu", ["Register Face", "Mark Attendance", "Live Recognition", "Attendance", "Delete Face"])

if menu == "Register Face":
    register_face()
elif menu == "Mark Attendance":
    upload = st.radio("What do you want to upload?:", ["Upload Video", "Upload Image"])
    if upload=="Upload Video":
        recognize_faces_vid()
    elif upload=="Upload Image":
        recognize_faces_img()
elif menu == "Live Recognition":
    live_face_recognition()
elif menu == "Delete Face":
    delete_face()
elif menu == "Attendance":
    display_attendance_records()


