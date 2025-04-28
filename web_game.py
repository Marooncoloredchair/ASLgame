from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import random
import time
import base64
import json
import sys
import traceback

app = Flask(__name__)

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

class ASLGame:
    def __init__(self):
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model('new_asl_model.h5')
            print("Model loaded successfully")
            
            # Test the model with a dummy input
            dummy_input = np.zeros((1, 79))
            prediction = self.model.predict(dummy_input, verbose=0)
            print("Model test prediction shape:", prediction.shape)
            print("Model test prediction sample:", prediction[0][:5])
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            sys.exit(1)
            
        self.score = 0
        self.current_word = ""
        self.current_letter_index = 0
        self.category = ""
        self.last_prediction_time = 0
        self.prediction_delay = 0.5
        self.confidence_threshold = 0.3  # Lowered threshold to see more predictions
        self.cap = None
        self.last_detected_letter = None
        self.last_confidence = 0
        self.letter_held_time = 0
        self.letter_hold_threshold = 1.0
        self.last_correct_letter = None
        self.is_processing = False
        self.confidence_history = []  # For smoothing
        self.confidence_history_size = 5

    def start_new_word(self):
        try:
            self.category = random.choice(list(categories.keys()))
            self.current_word = random.choice(categories[self.category])
            self.current_letter_index = 0
            self.last_correct_letter = None
            return {
                'category': self.category,
                'word': self.current_word,
                'current_letter': self.current_word[self.current_letter_index]
            }
        except Exception as e:
            print(f"Error in start_new_word: {str(e)}")
            return {'error': str(e)}

    def check_letter(self, predicted_letter):
        try:
            if self.is_processing:
                return {'status': 'waiting', 'detected_letter': self.last_detected_letter, 'confidence': self.last_confidence}

            self.is_processing = True
            current_time = time.time()
            
            if current_time - self.last_prediction_time < self.prediction_delay:
                self.is_processing = False
                return {'status': 'waiting', 'detected_letter': self.last_detected_letter, 'confidence': self.last_confidence}

            self.last_prediction_time = current_time
            target_letter = self.current_word[self.current_letter_index]
            
            if predicted_letter == target_letter and self.last_confidence > self.confidence_threshold:
                if self.last_correct_letter != target_letter:
                    self.letter_held_time = current_time
                    self.last_correct_letter = target_letter
                    self.is_processing = False
                    return {'status': 'holding', 'detected_letter': predicted_letter, 'confidence': self.last_confidence}
                
                if current_time - self.letter_held_time >= self.letter_hold_threshold:
                    self.current_letter_index += 1
                    self.last_correct_letter = None
                    if self.current_letter_index >= len(self.current_word):
                        self.score += 1
                        self.is_processing = False
                        return {'status': 'completed', 'score': self.score}
                    else:
                        self.is_processing = False
                        return {'status': 'correct', 'next_letter': self.current_word[self.current_letter_index]}
                else:
                    self.is_processing = False
                    return {'status': 'holding', 'detected_letter': predicted_letter, 'confidence': self.last_confidence}
            
            self.last_correct_letter = None
            self.is_processing = False
            return {'status': 'incorrect', 'detected_letter': self.last_detected_letter, 'confidence': self.last_confidence}
        except Exception as e:
            print(f"Error in check_letter: {str(e)}")
            self.is_processing = False
            return {'status': 'error', 'message': str(e)}

    def preprocess_landmarks(self, landmarks):
        try:
            # Extract 21 hand landmarks (x, y, z)
            lm = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
            # Calculate distances between key points
            thumb_tip = lm[4]
            index_tip = lm[8]
            middle_tip = lm[12]
            ring_tip = lm[16]
            pinky_tip = lm[20]
            wrist = lm[0]
            distances = [
                np.linalg.norm(thumb_tip - index_tip),
                np.linalg.norm(thumb_tip - middle_tip),
                np.linalg.norm(thumb_tip - ring_tip),
                np.linalg.norm(thumb_tip - pinky_tip),
                np.linalg.norm(index_tip - middle_tip),
                np.linalg.norm(middle_tip - ring_tip),
                np.linalg.norm(ring_tip - pinky_tip),
                np.linalg.norm(thumb_tip - wrist),
                np.linalg.norm(index_tip - wrist),
                np.linalg.norm(middle_tip - wrist),
                np.linalg.norm(ring_tip - wrist),
                np.linalg.norm(pinky_tip - wrist)
            ]
            # Calculate angles between fingers
            def calculate_angle(a, b, c):
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.arccos(cosine_angle)
            angles = [
                calculate_angle(thumb_tip, wrist, index_tip),
                calculate_angle(index_tip, wrist, middle_tip),
                calculate_angle(middle_tip, wrist, ring_tip),
                calculate_angle(ring_tip, wrist, pinky_tip)
            ]
            # Combine all features
            features = np.concatenate([
                lm.flatten(),
                distances,
                angles
            ])
            # Normalize features
            features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
            return features.reshape(1, -1)
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return np.zeros((1, 79))

# Create game instance
try:
    game = ASLGame()
except Exception as e:
    print(f"Failed to initialize game: {str(e)}")
    sys.exit(1)

def generate_frames():
    try:
        if game.cap is None:
            print("Initializing camera...")
            game.cap = cv2.VideoCapture(0)
            if not game.cap.isOpened():
                raise Exception("Could not open video device")
            print("Camera initialized successfully")

        while True:
            try:
                success, frame = game.cap.read()
                if not success:
                    print("Failed to read frame")
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        try:
                            features = game.preprocess_landmarks(hand_landmarks)
                            prediction = game.model.predict(features, verbose=0)
                            
                            # Get top 3 predictions
                            top_k = 3
                            top_indices = np.argsort(prediction[0])[-top_k:][::-1]
                            top_letters = [letters[i] for i in top_indices]
                            top_confidences = prediction[0][top_indices]
                            
                            # Print debug information
                            print("\nTop predictions:")
                            for letter, conf in zip(top_letters, top_confidences):
                                print(f"{letter}: {conf:.2%}")
                            
                            # Update game state with the highest confidence prediction
                            confidence = float(top_confidences[0])
                            predicted_letter = top_letters[0]
                            
                            game.last_detected_letter = predicted_letter
                            game.last_confidence = confidence
                            
                            # Update confidence history for smoothing
                            game.confidence_history.append(confidence)
                            if len(game.confidence_history) > game.confidence_history_size:
                                game.confidence_history.pop(0)
                            
                            # Draw detection info on frame
                            cv2.putText(frame, f'Letter: {predicted_letter}', 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f'Confidence: {confidence:.2%}', 
                                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Draw top 3 predictions
                            for i, (letter, conf) in enumerate(zip(top_letters, top_confidences)):
                                y_pos = 110 + i * 40
                                cv2.putText(frame, f'{letter}: {conf:.2%}', 
                                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                        except Exception as e:
                            print(f"Error in prediction: {str(e)}")
                            print(traceback.format_exc())
                            cv2.putText(frame, "Error in prediction", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    game.last_detected_letter = None
                    game.last_confidence = 0
                    game.confidence_history = []
                    cv2.putText(frame, "No hand detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in frame generation: {str(e)}")
                print(traceback.format_exc())
                continue
    except Exception as e:
        print(f"Fatal error in video feed: {str(e)}")
        print(traceback.format_exc())
        return

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_game')
def start_game():
    try:
        return jsonify(game.start_new_word())
    except Exception as e:
        print(f"Error in start_game: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/check_letter/<letter>')
def check_letter(letter):
    try:
        # Convert all float32 in the response to float
        result = game.check_letter(letter)
        if 'confidence' in result:
            result['confidence'] = float(result['confidence']) if result['confidence'] is not None else None
        if 'score' in result:
            result['score'] = float(result['score']) if result['score'] is not None else None
        return jsonify(result)
    except Exception as e:
        print(f"Error in check_letter: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/get_detected_letter')
def get_detected_letter():
    try:
        return jsonify({
            'letter': game.last_detected_letter,
            'confidence': float(game.last_confidence) if game.last_confidence is not None else None
        })
    except Exception as e:
        print(f"Error in get_detected_letter: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/game_status')
def game_status():
    try:
        # Compose atomic game state
        status = ''
        # Use the same logic as in check_letter, but don't advance the game
        detected_letter = game.last_detected_letter
        # Use moving average of confidence for smoothing
        if game.confidence_history:
            confidence = float(np.mean(game.confidence_history))
        else:
            confidence = float(game.last_confidence) if game.last_confidence is not None else None
        score = float(game.score)
        current_letter = game.current_word[game.current_letter_index] if game.current_word and game.current_letter_index < len(game.current_word) else ''
        category = game.category
        word = game.current_word
        # Determine status
        if detected_letter is not None and current_letter:
            if detected_letter == current_letter and confidence > game.confidence_threshold:
                if game.last_correct_letter == current_letter:
                    status = 'holding'
                else:
                    status = 'correct'
            else:
                status = 'incorrect'
        else:
            status = ''
        return jsonify({
            'detected_letter': detected_letter,
            'confidence': confidence,
            'status': status,
            'score': score,
            'current_letter': current_letter,
            'category': category,
            'word': word
        })
    except Exception as e:
        print(f"Error in game_status: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    try:
        print("Starting Flask server...")
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1) 