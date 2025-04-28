from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import random

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dictionary mapping class indices to letters
class_to_letter = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

# Try to load the improved model, fall back to the original if not available
try:
    model = tf.keras.models.load_model('improved_asl_model.h5')
    print("Loaded improved model")
except:
    model = tf.keras.models.load_model('asl_model.h5')
    print("Loaded original model")

# Game state
game_state = {
    'score': 0,
    'target_letter': None,
    'start_time': None,
    'last_prediction': None,
    'last_confidence': 0,
    'last_correct': False,
    'prediction_history': []
}

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Resize to 28x28
    resized = cv2.resize(enhanced, (28, 28))
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Reshape for the model
    reshaped = np.reshape(normalized, (28, 28, 1))
    
    return reshaped

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Create a larger display frame
        display_frame = cv2.resize(frame, (640, 480))
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand bounding box
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                
                # Add padding to the bounding box
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # Extract ROI
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size != 0:
                    processed_roi = preprocess_frame(roi)
                    prediction = model.predict(np.array([processed_roi]))
                    predicted_class = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class]
                    
                    # Update game state
                    game_state['last_prediction'] = class_to_letter[predicted_class]
                    game_state['last_confidence'] = float(confidence)
                    
                    # Check if prediction matches target letter
                    if game_state['target_letter'] and game_state['last_prediction'] == game_state['target_letter']:
                        if confidence > 0.7:  # Confidence threshold
                            if not game_state['last_correct']:
                                game_state['score'] += 1
                                game_state['last_correct'] = True
                                game_state['prediction_history'].append({
                                    'letter': game_state['last_prediction'],
                                    'confidence': game_state['last_confidence'],
                                    'correct': True
                                })
                    else:
                        game_state['last_correct'] = False
                    
                    # Draw hand landmarks and prediction
                    mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Display prediction and confidence
                    color = (0, 255, 0) if game_state['last_correct'] else (0, 0, 255)
                    cv2.putText(display_frame, f"Prediction: {game_state['last_prediction']}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(display_frame, f"Confidence: {game_state['last_confidence']:.2f}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Display target letter
                    if game_state['target_letter']:
                        cv2.putText(display_frame, f"Target: {game_state['target_letter']}", (10, 110), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display score and time
        if game_state['start_time']:
            elapsed_time = time.time() - game_state['start_time']
            remaining_time = max(0, 60 - elapsed_time)
            cv2.putText(display_frame, f"Score: {game_state['score']}", (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {remaining_time:.1f}s", (10, 190), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_game')
def start_game():
    game_state['score'] = 0
    game_state['target_letter'] = random.choice(list(class_to_letter.values()))
    game_state['start_time'] = time.time()
    game_state['last_correct'] = False
    game_state['prediction_history'] = []
    return jsonify({
        'target_letter': game_state['target_letter'],
        'score': game_state['score']
    })

@app.route('/get_state')
def get_state():
    if game_state['start_time']:
        elapsed_time = time.time() - game_state['start_time']
        remaining_time = max(0, 60 - elapsed_time)
        if remaining_time <= 0:
            return jsonify({
                'game_over': True,
                'final_score': game_state['score'],
                'prediction_history': game_state['prediction_history']
            })
    return jsonify({
        'target_letter': game_state['target_letter'],
        'score': game_state['score'],
        'last_prediction': game_state['last_prediction'],
        'last_confidence': game_state['last_confidence'],
        'last_correct': game_state['last_correct']
    })

if __name__ == '__main__':
    app.run(debug=True) 