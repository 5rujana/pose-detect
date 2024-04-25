import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Define color codes
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
END_COLOR = '\033[0m'
pose = input("Enter the pose: ")
if pose in ["squat", "pushups", "lunges", "shoulder_press", "lat_pulldowns", "jumping_jacks", "leg_presses", "mountain_climbers", "burpees", "high_knees", "situps", "planks", "benchpress", "bicep_curls", "lat_pulldowns", "lat_raises", "leg_presses", "plank", "shoulder_presses"]:
    image_path_one = f"data/{pose}_1.png"
    image_path_two = f"data/{pose}_2.png"
    image_path_three = f"data/{pose}_3.png"
else:
    print("Pose not found")

def get_landmarks(pose_image):
    image = cv2.imread(pose_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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

    return angle1, angle2 #returning angles


def compare_pose(user_landmarks, pose_landmarks, angle1, angle2, distance_weight=0.8, angle_weight=0.2):
    if user_landmarks is None or pose_landmarks is None:
        print("Error: Landmarks not detected.")
        return None
    
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
        
        print(f"User Landmark: ({user_lm.x:.2f}, {user_lm.y:.2f}, {user_lm.z:.2f})")
        print(f"Pose Landmark: ({pose_lm.x:.2f}, {pose_lm.y:.2f}, {pose_lm.z:.2f})")
        print(f"Distance: {distance:.2f}")

    # Calculate distance accuracy
    normalized_distances = [distance / max_possible_distance for distance in distances]
    distance_accuracy = 1 - np.mean(normalized_distances)

    # Calculate angle accuracy
    angle_accuracy = (1 - (abs(angle1) / 180.0)) * (1 - (abs(angle2) / 180.0))  # Angle accuracy as a product of angle deviations from 180 degrees

    # Combine distance and angle accuracies
    overall_accuracy = distance_weight * distance_accuracy + angle_weight * angle_accuracy

    return overall_accuracy * 100  # Ensure accuracy is within the range [0, 1]

#load reference images
pose_landmarks_one = get_landmarks(image_path_one)
pose_landmarks_two = get_landmarks(image_path_two)
pose_landmarks_three = get_landmarks(image_path_three)

reference_landmarks = [pose_landmarks_one, pose_landmarks_two, pose_landmarks_three]

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            accuracyone = compare_pose(results.pose_landmarks.landmark, pose_landmarks_one)
            accuracytwo = compare_pose(results.pose_landmarks.landmark, pose_landmarks_two)
            accuracythree = compare_pose(results.pose_landmarks.landmark, pose_landmarks_three)

            # Define colors based on accuracy ranges
            color_one = (0, 255, 0) if accuracyone >= 90 else (0, 255, 255) if 50 <= accuracyone < 90 else (0, 0, 255)
            color_two = (0, 255, 0) if accuracytwo >= 90 else (0, 255, 255) if 50 <= accuracytwo < 90 else (0, 0, 255)
            color_three = (0, 255, 0) if accuracythree >= 90 else (0, 255, 255) if 50 <= accuracythree < 90 else (0, 0, 255)

            print("Accuracy for Pose 1:", accuracyone)
            print("Accuracy for Pose 2:", accuracytwo)
            print("Accuracy for Pose 3:", accuracythree)
            # Draw accuracy on the frame
            cv2.putText(image, f"1: {accuracyone:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_one, 1, cv2.LINE_AA)
            cv2.putText(image, f"2: {accuracytwo:.2f}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_two, 1, cv2.LINE_AA)
            cv2.putText(image, f"3: {accuracythree:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_three, 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('MediaPipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
