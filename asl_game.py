import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import random
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

# Game categories and words
categories = {
    'Animals': ['CAT', 'DOG', 'BIRD', 'FISH', 'BEAR', 'LION', 'TIGER', 'EAGLE'],
    'Food': ['PIZZA', 'BURGER', 'SALAD', 'APPLE', 'BANANA', 'GRAPE', 'BREAD', 'CAKE'],
    'Colors': ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE', 'ORANGE', 'BLACK', 'WHITE'],
    'Simple Words': ['HELLO', 'LOVE', 'PEACE', 'HAPPY', 'SMILE', 'DREAM', 'HOPE', 'JOY']
}

def preprocess_landmarks(landmarks):
    """Convert hand landmarks to a normalized feature vector"""
    features = []
    for landmark in landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features).reshape(1, 1, -1)

class ASLGame:
    def __init__(self):
        # Load the trained model
        self.model = tf.keras.models.load_model('new_asl_model.h5')
        self.score = 0
        self.current_word = ""
        self.current_letter_index = 0
        self.category = ""
        self.game_over = False
        self.last_prediction_time = 0
        self.prediction_delay = 1.0  # Delay between predictions in seconds
        self.confidence_threshold = 0.7

    def start_new_word(self):
        """Select a new random word from a random category"""
        self.category = random.choice(list(categories.keys()))
        self.current_word = random.choice(categories[self.category])
        self.current_letter_index = 0
        print(f"\nNew word from {self.category}: {self.current_word}")
        print(f"Sign the letter: {self.current_word[self.current_letter_index]}")

    def check_letter(self, predicted_letter):
        """Check if the predicted letter matches the current letter"""
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_delay:
            return False

        self.last_prediction_time = current_time
        target_letter = self.current_word[self.current_letter_index]
        
        if predicted_letter == target_letter:
            self.current_letter_index += 1
            if self.current_letter_index >= len(self.current_word):
                self.score += 1
                print(f"\nCorrect! Word completed! Score: {self.score}")
                return True
            else:
                print(f"\nCorrect! Next letter: {self.current_word[self.current_letter_index]}")
        return False

    def run(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        self.start_new_word()
        
        while cap.isOpened() and not self.game_over:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = hands.process(rgb_frame)
            
            # Draw game information
            cv2.putText(frame, f'Category: {self.category}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Word: {self.current_word}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Score: {self.score}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Current Letter: {self.current_word[self.current_letter_index]}', (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw hand landmarks and make predictions
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Preprocess landmarks and make prediction
                    features = preprocess_landmarks(hand_landmarks)
                    prediction = self.model.predict(features)
                    confidence = np.max(prediction)
                    
                    if confidence > self.confidence_threshold:
                        predicted_letter = letters[np.argmax(prediction)]
                        # Display prediction
                        cv2.putText(frame, f'Detected: {predicted_letter} ({confidence:.1%})',
                                  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Check if the letter is correct
                        if self.check_letter(predicted_letter):
                            self.start_new_word()
                            # Add a small delay before starting the new word
                            time.sleep(1)
            
            # Display the frame
            cv2.imshow('ASL Letter Game', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nGame Over! Final Score: {self.score}")

if __name__ == '__main__':
    game = ASLGame()
    game.run() 