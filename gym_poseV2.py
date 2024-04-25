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

def get_pose():
    pose = "squat"
    return pose


def load_pose_images(pose_name):
    pose_image_paths = []
    for i in range(1, 4):
        image_path = f"data\{pose_name}_{i}.png"
        print(f"Loading image from path: {image_path}")
        if os.path.exists(image_path): #checking if image exists
            pose_image_paths.append(image_path) #appending image path to list
        else:
            print(f"Image not found: {image_path}")
    return pose_image_paths #returning list of image paths


def get_landmarks(pose_image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        pose_landmarks = pose.process(pose_image)
        if pose_landmarks:
            return pose_landmarks.pose_landmarks
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
        if isinstance(user_landmarks, np.ndarray) and isinstance(pose_landmarks, np.ndarray): #checking if user_landmarks and pose_landmarks are numpy arrays
            if len(user_landmarks) != len(pose_landmarks):
                print("Number of landmarks do not match")
                return None
        elif isinstance(user_landmarks, mp.framework.formats.landmark_pb2.LandmarkList) and isinstance(pose_landmarks, mp.framework.formats.landmark_pb2.LandmarkList):
            #mp.framework.formats.landmark_pb2.LandmarkList is a protobuf message type that represents a list of landmarks detected in an image by the MediaPipe Pose model
            if len(user_landmarks.landmark) != len(pose_landmarks.landmark):
                print("Number of landmarks do not match")
                return None
        else:
            print("Error: Incompatible types for user_landmarks and pose_landmarks.")
            return None
        
        distances = []
        max_possible_distance = 0  # Initialize max possible distance
        
        for user_lm, pose_lm in zip(user_landmarks.landmark if isinstance(user_landmarks, mp.framework.formats.landmark_pb2.LandmarkList) else user_landmarks,
                                    pose_landmarks.landmark if isinstance(pose_landmarks, mp.framework.formats.landmark_pb2.LandmarkList) else pose_landmarks):
            #zip() function returns an iterator of tuples where the first item in each passed iterator is paired together, and then the second item in each passed iterator are paired together etc.
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
pose_image_paths = load_pose_images(pose_name)

#get reference pose landmarks
# Load reference images and extract landmarks
# Load reference images and extract landmarks
reference_landmarks = []
for image_path in pose_image_paths:
    image = cv2.imread(image_path)
    if image is not None:
        if isinstance(image, np.ndarray):
            landmarks = get_landmarks(image)
            if landmarks is not None:
                reference_landmarks.append(landmarks)
            else:
                print(f"Error: No landmarks detected in pose image {image_path}.")
        else:
            print(f"Error: Failed to load pose image {image_path}. Invalid image type.")
    else:
        print(f"Error: Failed to load pose image {image_path}. Image not found.")

# Check if reference landmarks are loaded correctly
if not reference_landmarks:
    print("Error: No reference landmarks loaded.")


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image) #camera feed image

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extract landmarks
        try:
            if results.pose_landmarks: #checking if landmarks are detected in the camera feed image
                user_landmarks = results.pose_landmarks
                landmarks = results.pose_landmarks.landmark
                #get the coordinates of the landmarks
                #shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                #elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                #wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                #knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                #ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                #hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                
                # Compare user landmarks with reference landmarks
                accuracy_results = compare_pose(user_landmarks, [get_landmarks(pose_image) for pose_image in pose_image_paths])
                if accuracy_results:
                    for i, accuracy in enumerate(accuracy_results):
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

        # Load and resize all three reference pose images
        pose_images_resized = [cv2.resize(cv2.imread(pose_image_path), (frame.shape[1], frame.shape[0])) for pose_image_path in pose_image_paths]

        # Iterate through each reference pose image and draw landmarks
        for i, pose_image_resized in enumerate(pose_images_resized):
            # Get landmarks for the current reference pose image
            pose_landmarks = get_landmarks(pose_image_resized)
            
            # Check if landmarks are detected in the current reference pose image
            if pose_landmarks:
                # Draw landmarks on the reference pose image
                mp_drawing.draw_landmarks(
                    pose_image_resized, pose_landmarks, mp_pose.POSE_CONNECTIONS,
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

        cv2.imshow('MediaPipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
