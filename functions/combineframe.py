import cv2
import mediapipe as mp
import numpy as np
# Load and resize all three reference pose images
pose_images_resized = [cv2.resize(cv2.imread(pose_image_path), (frame.shape[1], frame.shape[0])) for pose_image_path in pose_images]

# Iterate through each reference pose image and draw landmarks
for i, pose_image_resized in enumerate(pose_images_resized):
    if pose_landmarks[i] and results.pose_landmarks:
        # Draw landmarks on the reference pose image
        mp_drawing.draw_landmarks(
            pose_image_resized, pose_landmarks[i], mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

        # Draw landmarks on the webcam feed image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Combine the frame and the reference pose image
        combined_image = cv2.hconcat([image, pose_image_resized])
        cv2.imshow('MediaPipe Feed', combined_image)
    else:
        print(f"Error: No landmarks detected in pose image {i + 1}.")

# Display the webcam feed with landmarks
cv2.imshow('MediaPipe Feed', image)

