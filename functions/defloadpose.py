import cv2
import mediapipe as mp
import numpy as np

def load_pose_images(pose_name):
    pose_images = []
    for i in range(1, 4):
        image_path = f"data/{pose_name}_{i}.jpg"
        image = cv2.imread(image_path)
        pose_images.append(image)
    return pose_images