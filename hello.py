import cv2 as cv
import mediapipe as mp

mp_pose = mp.solutions.pose # Import mediapipe pose module
pose = mp_pose.Pose() # Create a pose object

cap = cv.VideoCapture(0)
NOSE = 0
RIGHT_EYE_INNER = 1
RIGHT_EYE = 2
RIGHT_EYE_OUTER = 3
LEFT_EYE_INNER = 4
LEFT_EYE = 5
LEFT_EYE_OUTER = 6
RIGHT_EAR = 7
LEFT_EAR = 8
MOUTH_RIGHT = 9
MOUTH_LEFT = 10
RIGHT_SHOULDER = 11
LEFT_SHOULDER = 12
RIGHT_ELBOW = 13
LEFT_ELBOW = 14
RIGHT_WRIST = 15
LEFT_WRIST = 16
RIGHT_HIP = 23
LEFT_HIP = 24
RIGHT_KNEE = 25
LEFT_KNEE = 26
RIGHT_ANKLE = 27
LEFT_ANKLE = 28

while True:
    ret, frame = cap.read() # Read the frame from the webcam
    #ret is a boolean value that returns true if the frame is available
    if not ret:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)#converts  bgr filr to rgb
    #mideapipe's pose model expects rgb
    results = pose.process(frame_rgb)#process the rgb using mediapipe pose model

    if results.pose_landmarks: #if pose landmarks are detected
        landmarks = results.pose_landmarks.landmark #extracts the landmarks from the detected pose
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_hip = landmarks[RIGHT_HIP]
        left_hip = landmarks[LEFT_HIP]
        angle_shoulders = abs(right_shoulder.y - left_shoulder.y)
        angle_hips = abs(right_hip.y - left_hip.y)
        if angle_shoulders < angle_hips:
            posture_text = "Correct posture"
            color = (0, 255, 0)
        else:
            posture_text = "Incorrect posture"
            color = (0, 0, 255)
        cv.putText(frame, posture_text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.imshow("Gym Posture Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()  # Fix the typo in the function name
