import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define the ASL letters
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Create training data directory if it doesn't exist
training_dir = 'training_data'
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

# Create subdirectories for each letter
for letter in letters:
    letter_dir = os.path.join(training_dir, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

def preprocess_landmarks(landmarks):
    """Convert hand landmarks to a normalized feature vector"""
    features = []
    for landmark in landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    features = np.array(features)
    features = (features - np.min(features)) / (np.max(features) - np.min(features))
    return features  # Return as 1D array

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    current_letter_index = 0
    samples_collected = 0
    total_samples = 100
    capture_delay = 0.02  # Super short delay between captures
    last_capture_time = 0

    print("\nInstructions:")
    print("1. Show each letter sign clearly to the camera")
    print("2. Press 'n' to skip to next letter")
    print("3. Press 'b' to go back to previous letter")
    print("4. Press 'q' to quit")
    print("\nStarting with letter:", letters[current_letter_index])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and capture samples
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Auto-capture if enough time has passed
                current_time = time.time()
                if current_time - last_capture_time >= capture_delay:
                    # Capture the sample
                    features = preprocess_landmarks(hand_landmarks)
                    sample_path = os.path.join(
                        training_dir, 
                        letters[current_letter_index], 
                        f'sample_{samples_collected}.npy'
                    )
                    np.save(sample_path, features)
                    
                    samples_collected += 1
                    last_capture_time = current_time
                    
                    # Show progress
                    print(f"\rCollecting letter {letters[current_letter_index]}: {samples_collected}/{total_samples}", end='')
                    
                    # Move to next letter when enough samples collected
                    if samples_collected >= total_samples:
                        samples_collected = 0
                        current_letter_index = (current_letter_index + 1) % len(letters)
                        print(f"\nMoving to letter: {letters[current_letter_index]}")
                        time.sleep(1)  # Brief pause between letters
        
        # Display the current letter and sample count
        cv2.putText(frame, f'Show Letter: {letters[current_letter_index]}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Samples: {samples_collected}/{total_samples}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show if hand is detected
        if results.multi_hand_landmarks:
            cv2.putText(frame, 'Hand Detected - Capturing', (10, 110), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Hand Detected', (10, 110), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('ASL Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            samples_collected = 0
            current_letter_index = (current_letter_index + 1) % len(letters)
            print(f"\nSkipping to letter: {letters[current_letter_index]}")
        elif key == ord('b'):
            samples_collected = 0
            current_letter_index = (current_letter_index - 1) % len(letters)
            print(f"\nGoing back to letter: {letters[current_letter_index]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 