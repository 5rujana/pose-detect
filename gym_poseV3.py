import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define color codes
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
END_COLOR = '\033[0m'

def get_pose():
    pose = "squat"
    return pose

def load_pose_images(pose_name):
    pose_images = []
    for i in range(1, 4):
        image_path = f"data/{pose_name}_{i}.jpg"
        image = cv2.imread(image_path)
        pose_images.append(image)
    return pose_images

def get_landmarks(pose_image):
    image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        else:
            return None

def calculate_angle(shoulder, elbow, wrist, knee, ankle, hip):
    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    wrist = np.array(wrist)
    knee = np.array(knee)
    ankle = np.array(ankle)
    hip = np.array(hip)

    angle1 = np.degrees(np.arctan2(wrist[1]-elbow[1], wrist[0]-elbow[0]) - np.arctan2(shoulder[1]-elbow[1], shoulder[0]-elbow[0]))
    angle2 = np.degrees(np.arctan2(ankle[1]-knee[1], ankle[0]-knee[0]) - np.arctan2(hip[1]-knee[1], hip[0]-knee[0]))

    return angle1, angle2


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

#get reference pose landmarks
reference_landmarks = [get_landmarks(pose_image) for pose_image in pose_images]

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extract landmarks
        try:
            if results.pose_landmarks:
                user_landmarks = results.pose_landmarks
                
                # Compare user landmarks with reference landmarks
                for i, reference_landmark in enumerate(reference_landmarks):
                    accuracy = compare_pose(user_landmarks.landmark, reference_landmark)
                    if accuracy is not None:
                        print(f"Accuracy for Pose {i + 1}: {accuracy:.2f}")
                        
                        # Check accuracy and print in respective color
                        if accuracy < 50:
                            print(RED + f"Accuracy for Pose {i + 1}: {accuracy:.4f}" + END_COLOR)
                        elif 50 <= accuracy < 90:
                            print(YELLOW + f"Accuracy for Pose {i + 1}: {accuracy:.4f}" + END_COLOR)
                        else:
                            print(GREEN + f"Accuracy for Pose {i + 1}: {accuracy:.4f}" + END_COLOR)

                        # Conditions to present accuracy text with specified color on frame
                        if accuracy < 50:
                            color = (0, 0, 255)
                        elif 50 <= accuracy < 90:
                            color = (0, 255, 255)
                        else:
                            color = (0, 255, 0)
                        
                        cv2.putText(image, f"Accuracy for Pose {i + 1}: {accuracy:.2f}", (50, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error: {e}")

        # Draw user landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Display the webcam feed with landmarks
        cv2.imshow('MediaPipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
