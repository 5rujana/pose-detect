import cv2
import mediapipe as mp
import numpy as np
def compare_pose(user_landmarks, pose_images):
    if user_landmarks is None or not pose_images:
        print("Error: Landmarks not detected.")
        return None

    pose_landmarks_list = []
    for pose_image in pose_images:
        pose_landmarks = get_landmarks(pose_image)
        if pose_landmarks:
            pose_landmarks_list.append(pose_landmarks)
        else:
            print("Error: Landmarks not detected in reference pose image.")
    
    accuracy_results = []
    for pose_landmarks in pose_landmarks_list:
        if len(user_landmarks) != len(pose_landmarks):
            print("Number of landmarks do not match")
            return None
        
        distances = []
        max_possible_distance = 0  # Initialize max possible distance
        
        for user_lm, pose_lm in zip(user_landmarks, pose_landmarks):
            user_point = np.array([user_lm.x, user_lm.y, user_lm.z])
            pose_point = np.array([pose_lm.x, pose_lm.y, pose_lm.z])
            distance = np.linalg.norm(user_point - pose_point)
            distances.append(distance)
            
            # Update max possible distance if the current distance is greater
            max_possible_distance = max(max_possible_distance, distance)
        
        # Calculate accuracy as the mean normalized distance between corresponding landmarks
        normalized_distances = [distance / max_possible_distance for distance in distances]
        accuracy = 1 - np.mean(normalized_distances)
        accuracy_results.append(accuracy * 100)
    
    return accuracy_results

# Get user's desired pose
pose_name = get_pose()
pose_images = load_pose_images(pose_name)
