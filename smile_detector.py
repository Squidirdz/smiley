import cv2
import pyttsx3
import datetime
import os
import random
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageTk
import tkinter as tk

# === Setup Folders ===
if not os.path.exists("happy_moments"):
    os.makedirs("happy_moments")

# === Text-to-Speech ===
engine = pyttsx3.init()
engine.setProperty('rate', 160)
def speak(msg):
    print("[AI]:", msg)
    engine.say(msg)
    engine.runAndWait()

# === Mediapipe FaceMesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# === Positive Messages ===
positive_messages = [
    "Your smile lights up the world!",
    "You look amazing when you smile!",
    "Your smile is contagious! Keep it up!",
    "You're glowing with happiness!",
    "A smile can change everything—yours is perfect!",
    "Your smile brightens up the room!",
    "Keep smiling, the world needs more of it!",
]

# === GUI Setup ===
window = tk.Tk()
window.title("Smile Detector")
window.geometry("800x600")
window.configure(bg='black')

# Add text above the animation
label = tk.Label(window, text="Smile Assistant 😄", font=("Arial", 24), fg="white", bg="black")
label.pack(pady=10)

# --- Emoji Animation Setup ---
# Define emojis for sad, neutral, and happy faces
sad_emoji = "😞"
happy_emoji = "😄"

# Create label for emoji animation
animation_label = tk.Label(window, text=sad_emoji, font=("Arial", 100), fg="white", bg="black")
animation_label.pack(pady=50)

# Create label for status updates
status_label = tk.Label(window, text="Show me a smile, teeth, or tongue! 😁", font=("Arial", 14), fg="white", bg="black")
status_label.pack(pady=10)

# === Webcam ===
cap = cv2.VideoCapture(0)

smile_detected = False

# === Mouth Aspect Ratio ===
def mouth_aspect_ratio(landmarks):
    top_lip = np.array([landmarks[13].y, landmarks[13].x])
    bottom_lip = np.array([landmarks[14].y, landmarks[14].x])
    left = np.array([landmarks[61].y, landmarks[61].x])
    right = np.array([landmarks[291].y, landmarks[291].x])

    vertical = np.linalg.norm(top_lip - bottom_lip)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0: return 0
    return vertical / horizontal

# === Smile or Mouth Open Detection ===
def detect_smile_or_open_mouth(frame, landmarks):
    mar = mouth_aspect_ratio(landmarks)
    if mar >= 0.35:
        return "mouth_open"

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        if emotion == "happy":
            return "smile"
    except Exception as e:
        print("Emotion detection failed:", e)

    return None

# === Frame Processing ===
def process_frame():
    global smile_detected

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    smile_type = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        smile_type = detect_smile_or_open_mouth(rgb_frame, face_landmarks)

    if not smile_detected:
        if smile_type:
            smile_detected = True
            # Immediately change to the happy emoji
            update_animation(happy_emoji)
            status_label.config(text="Your smile is amazing!")

            # Capture the photo and save it (after showing happy emoji)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"happy_moments/smile_{timestamp}.jpg"
            cv2.imwrite(path, frame)

            # Add delay to show the happy emoji before speaking
            window.after(500, speak_message)  # Delay before AI speaks

    # Keep updating the frame to detect smile
    window.after(10, process_frame)  # Delay to keep refreshing

# --- Update Emoji Animation ---
def update_animation(emoji):
    animation_label.config(text=emoji)

# === Speech after detection ===
def speak_message():
    message = random.choice(positive_messages)
    speak(message)
    
    # After AI speaks, say "Have a nice day!"
    speak("Have a nice day!")
    
    # After speech, quit the program
    window.after(2000, quit_program)  # 2 seconds delay before quitting

# === Quit Program ===
def quit_program():
    window.destroy()
    cap.release()
    cv2.destroyAllWindows()

# === Delayed Start After GUI Appears ===
def start_assistant():
    speak("Hello! Show me a smile, your teeth, or stick your tongue out!")
    process_frame()

# Start GUI and delay speech
window.after(1000, start_assistant)
window.mainloop()
cap.release()
cv2.destroyAllWindows()
