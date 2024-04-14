import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Function to get pose
def get_pose():
    pose = "demo"
    return pose # For demonstration purposes, replace with actual pose detection logic
# Load the reference pose image and extract landmarks
def get_landmarks(pose_image):
    image = cv2.imread(pose_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        else:
            return None

# Calculate angle between joints
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

# Compare user pose with reference pose image
def compare_pose(user_landmarks, pose_landmarks):
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
    

    # Calculate accuracy as the mean normalized distance between corresponding landmarks
    normalized_distances = [distance / max_possible_distance for distance in distances]
    accuracy = 1 - np.mean(normalized_distances)

    
    return accuracy*100  # Ensure accuracy is within the range [0, 1]




# Define image paths for each pose
squats_image_path = "data/Squat.jpg" #4
pushups_image_path = "data/pushups.jpg"
lunges_image_path = "data/lunges.jpg"
shoulder_press_image_path = "data/shoulder_presses.jpg" #1
bicep_curls_image_path = "data/bicep_curls.jpg"#6
lat_raises_image_path = "data/lat_raises.jpg"
lat_pulldowns_image_path = "data/lat_pulldowns.jpg" #5
leg_presses_image_path = "data/leg_presses.jpg" #3
jumping_jacks_image_path = "data/jumping_jacks.jpg"
leg_raises_image_path = "data/leg_raises.jpg"
mountain_climbers_image_path = "data/mountain_climbers.jpg"
burpees_image_path = "data/burpees.jpg"
high_knees_image_path = "data/high_knees.jpg"
bench_press_image_path = "data/benchpress.jpg"#7
situps_image_path = "data/situps.jpg"
planks_image_path = "data/plank.jpg"#2
demo_image_path= "data\demo.png"

# Get user's desired pose
pose_name = get_pose()
pose_image_path = None

if pose_name == "squats":
    pose_image_path = squats_image_path
elif pose_name == "pushups":
    pose_image_path = pushups_image_path
elif pose_name == "lunges":
    pose_image_path = lunges_image_path
elif pose_name == "shoulder_press":
    pose_image_path = shoulder_press_image_path
elif pose_name == "bicep_curls":
    pose_image_path = bicep_curls_image_path
elif pose_name == "lat_raises":
    pose_image_path = lat_raises_image_path
elif pose_name == "lat_pulldowns":
    pose_image_path = lat_pulldowns_image_path
elif pose_name == "leg_presses":
    pose_image_path = leg_presses_image_path
elif pose_name == "jumping_jacks":
    pose_image_path = jumping_jacks_image_path
elif pose_name == "leg_raises":
    pose_image_path = leg_raises_image_path
elif pose_name == "mountain_climbers":
    pose_image_path = mountain_climbers_image_path
elif pose_name == "burpees":
    pose_image_path = burpees_image_path
elif pose_name == "high_knees":
    pose_image_path = high_knees_image_path
elif pose_name == "situps":
    pose_image_path = situps_image_path
elif pose_name == "planks":
    pose_image_path = planks_image_path
elif pose_name == "bench_press":
    pose_image_path = bench_press_image_path
elif pose_name == "demo":
    pose_image_path = demo_image_path

if pose_image_path:
    # Load reference pose image
    pose_landmarks = get_landmarks(pose_image_path)


cap = cv2.VideoCapture(0)
#set up mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()

        # recolor our image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Image is no longer writeable
        #make detection
        results = pose.process(image)
        #recolor image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #get the coordinates of the landmarks
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            
            #calculate the angle
            angle = calculate_angle(shoulder, elbow, wrist, knee, ankle, hip)
            
            # Compare user pose with reference pose
            accuracy = compare_pose(results.pose_landmarks.landmark, pose_landmarks)
            
            # Define color codes
            RED = '\033[91m'
            YELLOW = '\033[93m'
            GREEN = '\033[92m'
            END_COLOR = '\033[0m'

            # Check accuracy and print in respective color
            if accuracy < 50:
                print(RED + "Accuracy: {:.4f}".format(accuracy) + END_COLOR)
            elif 50 <= accuracy < 90:
                print(YELLOW + "Accuracy: {:.4f}".format(accuracy) + END_COLOR)
            else:
                print(GREEN + "Accuracy: {:.4f}".format(accuracy) + END_COLOR)

            #fconditions to present accuracy text with specified color on frame
            if accuracy<50:
                color = (0,0,255)
            elif 50<= accuracy<90:
                color = (0,255,255)
            else:
                color = (0,255,0)
            
            # Display angle and accuracy
            #cv2.putText(image, f"Angle: {angle}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Accuracy: {accuracy:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        #render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  #change the color of the joints and connections
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), #for the joints
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) #for the connections

                                  
                                  )

        cv2.imshow('MediaPipe Feed',image) 

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

