import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

weapon_model = load_model("object_classifier_cv.h5")

classes = ["gun", "knife", "smartphone", "credit_card"]

def preprocess_frame(frame, img_height=150, img_width=150):
    """Preprocess the frame for classification."""
    image = cv2.resize(frame, (img_height, img_width))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def calculate_face_coverage(face_landmarks, brightness):
    """Calculate face coverage based on critical landmarks."""
    critical_landmarks = [33, 133, 362, 263, 1, 2, 98, 324, 13, 14]
    total_landmarks = len(critical_landmarks)
    visible_landmarks = 0

    visibility_threshold = 0.6 - (brightness / 255.0) * 0.2
    z_threshold_min = -0.1 - (brightness / 255.0) * 0.05
    z_threshold_max = 0.1 + (brightness / 255.0) * 0.05

    for idx in critical_landmarks:
        landmark = face_landmarks[idx]
        if landmark.visibility > visibility_threshold and z_threshold_min < landmark.z < z_threshold_max:
            visible_landmarks += 1

    coverage = (visible_landmarks / total_landmarks) * 100
    return coverage

def analyze_behavior(pose_landmarks):
    """Analyze unusual behavior based on hand positions."""
    left_hand_y = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_hand_y = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
    nose_y = pose_landmarks[mp_pose.PoseLandmark.NOSE].y

    if left_hand_y < nose_y or right_hand_y < nose_y:
        return True
    return False

#cap = cv2.VideoCapture("sample2.mp4")
cap = cv2.VideoCapture(0)
alert_message = "Warning: Potential Threat!"
alert_duration = 3
last_alert_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(image_rgb)
    pose_results = pose.process(image_rgb)

    unusual_behavior_detected = False

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            coverage = calculate_face_coverage(face_landmarks.landmark, brightness)
            print(f"Coverage: {coverage:.2f}% (Brightness: {brightness:.2f})")

            if coverage < 20:
                current_time = time.time()
                if current_time - last_alert_time > alert_duration:
                    cv2.putText(frame, alert_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"warning_image_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    last_alert_time = current_time

    if pose_results.pose_landmarks:
        unusual_behavior_detected = analyze_behavior(pose_results.pose_landmarks.landmark)

    if unusual_behavior_detected:
        cv2.putText(frame, "Unusual Behavior Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"behavior_warning_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

    preprocessed_frame = preprocess_frame(frame)
    predictions = weapon_model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    if predicted_class in ["knife", "gun"] and confidence > 0.8:
        cv2.putText(frame, f"Weapon Detected: {predicted_class} ({confidence:.2f})", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"weapon_detected_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

    cv2.imshow('Surveillance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
