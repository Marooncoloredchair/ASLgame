import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Define the ASL letters
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def augment_data(landmarks):
    """Apply data augmentation to landmarks"""
    augmented = []
    # Original data
    augmented.append(landmarks)
    
    # Add small random noise
    noise = np.random.normal(0, 0.01, landmarks.shape)
    augmented.append(landmarks + noise)
    
    # Add small rotation
    angle = np.random.uniform(-5, 5)
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])
    rotated = np.dot(landmarks.reshape(-1, 2), rotation_matrix)
    augmented.append(rotated.reshape(landmarks.shape))
    
    return np.array(augmented)

def preprocess_frame(frame):
    """Preprocess a single frame"""
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x, y, z coordinates
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    return None

def create_model(input_shape, num_classes):
    """Create an enhanced CNN model for image classification"""
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Loading training data...")
    # Load the existing training data
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
    
    print(f"Loaded {len(X)} training samples")
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    print("Preparing training data...")
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    y_train = np.eye(len(letters))[y_train]
    y_val = np.eye(len(letters))[y_val]
    
    print("Creating and training model...")
    # Create and train model
    model = create_model((64, 64, 1), len(letters))
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        ModelCheckpoint(
            'asl_model_enhanced.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("Saving model...")
    model.save('asl_model_enhanced.h5')
    
    # Print final accuracy
    print("Evaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    main() 