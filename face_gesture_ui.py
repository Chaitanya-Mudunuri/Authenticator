import cv2
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import mediapipe as mp
from mtcnn import MTCNN

# Initialize face detector
face_detector = MTCNN()

# Initialize hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables
running = False
cap = None

# Function to recognize faces
def recognize_faces(frame):
    result = face_detector.detect_faces(frame)
    if result:
        for face in result:
            bounding_box = face['box']
            cv2.rectangle(frame, 
                          (bounding_box[0], bounding_box[1]), 
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), 
                          (0, 255, 0), 2)
    return frame

# Function to recognize hand gestures
def recognize_hand_gesture(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    recognized_gesture = "Unknown"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            recognized_gesture = "Detected"  # Replace with actual gesture classification
    return frame, recognized_gesture

# Function to process video frames
def process_frame():
    global running, cap
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Call recognition functions
        frame = recognize_faces(frame)
        frame, recognized_gesture = recognize_hand_gesture(frame)
        
        # Convert frame for Tkinter display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Update label with new image
        lbl_video.imgtk = img
        lbl_video.configure(image=img)
        lbl_video.after(10, process_frame)

# Start recognition
def start_recognition():
    global running, cap
    running = True
    cap = cv2.VideoCapture(0)
    thread = threading.Thread(target=process_frame)
    thread.start()

# Stop recognition
def stop_recognition():
    global running, cap
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()

# Initialize UI
root = tk.Tk()
root.title("Face & Hand Gesture Recognition")
root.geometry("800x600")

# Video Label
lbl_video = Label(root)
lbl_video.pack()

# Start Button
btn_start = Button(root, text="Start Recognition", command=start_recognition)
btn_start.pack()

# Stop Button
btn_stop = Button(root, text="Stop Recognition", command=stop_recognition)
btn_stop.pack()

# Run Tkinter loop
root.mainloop()
