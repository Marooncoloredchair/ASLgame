import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import os
import json

# Define ASL letters
ASL_LETTERS = [chr(i) for i in range(ord('A'), ord('Z')+1)]

def load_kaggle_data(data_dir):
    """Load processed Kaggle dataset"""
    print("Loading Kaggle dataset...")
    X = []
    y = []
    class_counts = {}
    
    for letter in ASL_LETTERS:
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.exists(letter_dir):
            print(f"Warning: Directory {letter_dir} does not exist")
            continue
            
        samples = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
        class_counts[letter] = len(samples)
        
        print(f"Loading {len(samples)} samples for letter {letter}")
        for sample_file in samples:
            sample_path = os.path.join(letter_dir, sample_file)
            try:
                features = np.load(sample_path)
                X.append(features)
                y.append(ASL_LETTERS.index(letter))
            except Exception as e:
                print(f"Error loading {sample_path}: {e}")
    
    if not X:
        print("Error: No training data loaded!")
        return None, None
        
    X = np.array(X)
    y = np.array(y)
    
    print("\nClass distribution:")
    for letter, count in class_counts.items():
        print(f"{letter}: {count} samples")
    
    print(f"\nLoaded {len(X)} training samples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

def create_model(input_shape):
    """Create a CNN-LSTM hybrid model"""
    print(f"\nCreating model with input shape: {input_shape}")
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Reshape for CNN
    x = Reshape((input_shape[0], 1))(input_layer)
    
    # CNN layers
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # LSTM layers
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    output_layer = Dense(26, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def lr_schedule(epoch):
    """Learning rate schedule with warmup"""
    initial_lr = 0.001
    if epoch < 5:
        return initial_lr * (epoch + 1) / 5  # Warmup
    elif epoch < 15:
        return initial_lr
    elif epoch < 30:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

def main():
    print("Starting training process...")
    
    # Load Kaggle dataset
    kaggle_data_dir = "kaggle_processed_data"
    if not os.path.exists(kaggle_data_dir):
        print("Error: Kaggle dataset not found. Please run download_kaggle_data.py first.")
        return
    
    X, y = load_kaggle_data(kaggle_data_dir)
    if X is None or y is None:
        print("Failed to load Kaggle dataset. Exiting.")
        return
    
    # Reshape data for model
    X = X.reshape(X.shape[0], -1)  # Flatten to (samples, features)
    print(f"Reshaped X to: {X.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create model
    model = create_model(input_shape=(X.shape[1],))
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint('new_asl_model.h5', monitor='val_accuracy', save_best_only=True),
        LearningRateScheduler(lr_schedule)
    ]
    
    print("\nStarting training...")
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 