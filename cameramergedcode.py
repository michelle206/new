import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose class and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to check if the person is lying down
def is_lying_down(landmarks):
    # Extract the y-coordinates of important body points
    points_of_interest = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    y_coords = [landmarks[mp_pose.PoseLandmark[point].value].y for point in points_of_interest]
    
    # Calculate the range of the y-coordinates
    y_range = np.ptp(y_coords)  # Peak-to-peak (max - min)
    
    # Set a threshold to determine if the person is lying down
    return y_range < 0.2  # Adjust threshold as needed

def main():
    st.set_page_config(page_title="Webcam Pose Estimation App")
    st.title("Webcam Pose Estimation Streamlit App")
    st.caption("Powered by OpenCV and Mediapipe")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    # Set up Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()

            if not ret:
                st.write("Video Capture Ended")
                break
            
            # Convert the frame to RGB as Mediapipe requires
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # Improve performance

            # Make pose detection
            results = pose.process(image_rgb)

            # Draw pose landmarks on the image
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

                # Check if the person is lying down
                if is_lying_down(results.pose_landmarks.landmark):
                    cv2.putText(image_bgr, "Lying Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image_bgr, "Not Lying Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the image with pose landmarks
            frame_placeholder.image(image_bgr, channels="BGR")

            # Break the loop on 'q' key press (handled in the Streamlit interface)
            if st.button("Quit"):
                break

    # Release the camera
    cap.release()

if __name__ == "__main__":
    main()
