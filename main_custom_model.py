import os
import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound  
import threading  

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

weapon_model = load_model("knife_classifier_model.h5")

classes = ["not_knife", "knife"]

SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

ALARM_SOUND = "alarm.wav"  
alarm_playing = False  

def preprocess_frame(frame, img_height=224, img_width=224):
    """Preprocess the frame for classification."""
    image = cv2.resize(frame, (img_height, img_width))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, 
                              text_color=(255, 255, 255), background_color=(0, 0, 255), thickness=1):
    """Draw text with a background rectangle."""
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position

    cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w + 10, y + 5), background_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def save_detected_frame(frame, label):
    """Save the current frame with a unique filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{SAVE_DIR}/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved image: {filename}")

def play_alarm():
    """Play the alarm sound in a separate thread."""
    global alarm_playing
    if not alarm_playing:  
        alarm_playing = True
        playsound(ALARM_SOUND)
        alarm_playing = False

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(image_rgb)
    unusual_behavior_detected = False

    if pose_results.pose_landmarks:
        left_hand_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_hand_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        nose_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y

        if left_hand_y < nose_y or right_hand_y < nose_y:
            unusual_behavior_detected = True
            frame_height, frame_width, _ = frame.shape
            draw_text_with_background(frame, "Unusual Behavior Detected!", 
                                      position=(10, frame_height - 30), font_scale=0.7, 
                                      background_color=(255, 0, 0))
            save_detected_frame(frame, "unusual_behavior")

    preprocessed_frame = preprocess_frame(frame)
    prediction = weapon_model.predict(preprocessed_frame)[0][0]  
    predicted_class = classes[1] if prediction > 0.5 else classes[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    if predicted_class == "knife" and confidence >= 0.9:
        
        threading.Thread(target=play_alarm, daemon=True).start()
        draw_text_with_background(frame, f"Weapon Detected: {predicted_class}", (10, 30), font_scale=0.7)
        draw_text_with_background(frame, f"Confidence: {confidence:.2f}", (10, 60), font_scale=0.7)
        save_detected_frame(frame, "weapon_detected")

    cv2.imshow('Surveillance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
