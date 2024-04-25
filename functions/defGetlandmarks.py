import cv2
import mediapipe as mp
import numpy as np
def get_landmarks(pose_image):
    image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        else:
            return None