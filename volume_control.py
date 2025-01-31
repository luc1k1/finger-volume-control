import cv2
import mediapipe as mp
import numpy as np
import math

# For Windows volume control not for mac
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hand mode
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize volume control (Windows)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range for mapping
vol_min, vol_max = volume.GetVolumeRange()[:2]

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image for mirror effect
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract coordinates of thumb tip and index finger tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized coordinates to pixel values
            h, w, _ = img.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Draw circles on thumb and index finger tips
            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)

            # Draw line between thumb and index finger
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)

            # Calculate distance
            distance = calculate_distance((thumb_x, thumb_y), (index_x, index_y))

            # Map the distance to volume range
            # Define minimum and maximum distances
            min_dist = 30
            max_dist = 300
            distance = np.clip(distance, min_dist, max_dist)
            vol = np.interp(distance, [min_dist, max_dist], [vol_min, vol_max])

            # Set system volume (Windows)
            volume.SetMasterVolumeLevel(vol, None)

            # Optional: Display volume percentage
            vol_percent = np.interp(vol, [vol_min, vol_max], [0, 100])
            cv2.putText(img, f'Volume: {int(vol_percent)} %', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Volume Control', img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
