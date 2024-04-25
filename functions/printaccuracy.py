import cv2
import mediapipe as mp
import numpy as np
accuracy_results = compare_pose(user_landmarks, [get_landmarks(pose_image) for pose_image in pose_images])
if accuracy_results:
    for i, accuracy in enumerate(accuracy_results):
        print(f"Accuracy for Pose {i + 1}: {accuracy:.2f}")
        
        # Define color codes
        RED = '\033[91m'
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        END_COLOR = '\033[0m'

        # Check accuracy and print in respective color
        if accuracy < 50:
            print(RED + f"Accuracy for Pose {i + 1}: {accuracy:.4f}".format(accuracy) + END_COLOR)
        elif 50 <= accuracy < 90:
            print(YELLOW + f"Accuracy for Pose {i + 1}: {accuracy:.4f}".format(accuracy) + END_COLOR)
        else:
            print(GREEN + f"Accuracy for Pose {i + 1}: {accuracy:.4f}".format(accuracy) + END_COLOR)

        # Conditions to present accuracy text with specified color on frame
        if accuracy < 50:
            color = (0, 0, 255)
        elif 50 <= accuracy < 90:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)
        
        cv2.putText(image, f"Accuracy for Pose {i + 1}: {accuracy:.2f}", (50, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
