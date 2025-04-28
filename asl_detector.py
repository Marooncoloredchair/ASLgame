import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import json
from datetime import datetime

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

# Define letter-specific confidence thresholds
letter_thresholds = {
    'A': 0.5,  # Adjusted thresholds
    'B': 0.5,
    'C': 0.5,
    'D': 0.5,
    'E': 0.5,
    'F': 0.5,
    'G': 0.5,
    'H': 0.5,
    'I': 0.5,
    'J': 0.5,
    'K': 0.5,
    'L': 0.5,
    'M': 0.5,
    'N': 0.5,
    'O': 0.5,
    'P': 0.5,
    'Q': 0.5,
    'R': 0.5,
    'S': 0.5,
    'T': 0.5,
    'U': 0.5,
    'V': 0.5,
    'W': 0.5,
    'X': 0.5,
    'Y': 0.5,
    'Z': 0.5
}

# Create feedback directory if it doesn't exist
feedback_dir = 'feedback_data'
if not os.path.exists(feedback_dir):
    os.makedirs(feedback_dir)

def preprocess_landmarks(landmarks):
    """Convert hand landmarks to a normalized feature vector"""
    # Extract landmarks in a consistent order
    features = []
    for landmark in landmarks.landmark:
        # Ensure we get x, y, z in the same order for each landmark
        features.extend([landmark.x, landmark.y, landmark.z])
    
    if len(features) != 63:
        print(f"Warning: Expected 63 features but got {len(features)}")
        return None
    
    # Convert to numpy array and normalize
    features = np.array(features)
    
    # Normalize features to [0, 1] range
    min_val = np.min(features)
    max_val = np.max(features)
    if max_val != min_val:  # Avoid division by zero
        features = (features - min_val) / (max_val - min_val)
    else:
        features = np.zeros_like(features)  # If all values are the same
    
    return features.reshape(1, 1, -1)

def save_feedback(landmarks, predicted_letter, is_correct, actual_letter=None):
    """Save feedback data for future training"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_data = {
        'timestamp': timestamp,
        'landmarks': landmarks.tolist(),
        'predicted_letter': predicted_letter,
        'is_correct': is_correct,
        'actual_letter': actual_letter
    }
    
    feedback_file = os.path.join(feedback_dir, f'feedback_{timestamp}.json')
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f)
    
    print(f"Feedback saved to {feedback_file}")

def main():
    # Load the trained model
    try:
        model = tf.keras.models.load_model('new_asl_model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Instructions for user
    print("\nInstructions:")
    print("1. Press 'y' if the prediction is correct")
    print("2. Press 'n' if the prediction is wrong")
    print("3. Press 'q' to quit")
    print("4. Press 'c' to enter the correct letter when prediction is wrong")
    
    # Variables for smoothing predictions
    prediction_history = []
    history_size = 5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and make predictions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Preprocess landmarks and make prediction
                    features = preprocess_landmarks(hand_landmarks)
                    if features is None:
                        continue
                    prediction = model.predict(features)
                    confidence = np.max(prediction)
                    predicted_letter = letters[np.argmax(prediction)]
                    
                    # Add to prediction history
                    prediction_history.append(predicted_letter)
                    if len(prediction_history) > history_size:
                        prediction_history.pop(0)
                    
                    # Get the most common prediction from history
                    if prediction_history:
                        final_prediction = max(set(prediction_history), key=prediction_history.count)
                        final_confidence = confidence
                        
                        # Get the threshold for this letter
                        threshold = letter_thresholds.get(final_prediction, 0.5)
                        
                        # Only show prediction if confidence is above letter-specific threshold
                        if final_confidence > threshold:
                            cv2.putText(frame, f'Letter: {final_prediction} ({final_confidence:.1%})',
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, 'Press y/n for feedback',
                                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame, 'Not confident enough',
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error processing hand: {e}")
                    continue
        
        # Display the frame
        cv2.imshow('ASL Letter Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('y') and 'final_prediction' in locals():
            save_feedback(features, final_prediction, True)
            print("Correct prediction recorded!")
        elif key == ord('n') and 'final_prediction' in locals():
            print("Enter the correct letter (A-Z): ")
            actual_letter = input().upper()
            if actual_letter in letters:
                save_feedback(features, final_prediction, False, actual_letter)
                print("Incorrect prediction recorded!")
            else:
                print("Invalid letter entered!")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 