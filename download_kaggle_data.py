import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import shutil
import zipfile

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def preprocess_landmarks(landmarks):
    """Enhanced preprocessing of hand landmarks"""
    # Convert to numpy array
    landmarks = np.array(landmarks)
    
    # Reshape to (21, 3) for hand landmarks
    landmarks = landmarks.reshape(21, 3)
    
    # Calculate additional features
    # 1. Distances between key points
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
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
    
    # 2. Angles between fingers
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
        landmarks.flatten(),  # Original landmarks
        distances,           # Distances
        angles              # Angles
    ])
    
    # Normalize features
    features = (features - np.min(features)) / (np.max(features) - np.min(features))
    
    return features

def process_image(image_path):
    """Process a single image to extract hand landmarks"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmarks
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Preprocess landmarks
        features = preprocess_landmarks(landmarks)
        return features
    
    return None

def process_dataset():
    """Process the local ASL dataset"""
    print("Processing ASL dataset...")
    
    # Create temporary directory
    temp_dir = "temp_dataset"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # Extract the zip file
        print("Extracting dataset...")
        with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Create output directory for processed data
        output_dir = "kaggle_processed_data"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Create subdirectories for each letter
        letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        for letter in letters:
            os.makedirs(os.path.join(output_dir, letter))
        
        # Process images
        print("Processing images...")
        sample_counts = {letter: 0 for letter in letters}
        
        # Walk through the dataset directory
        for root, dirs, files in os.walk(temp_dir):
            for file in tqdm(files, desc="Processing images"):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    # Get the letter from the directory name
                    letter = os.path.basename(root).upper()
                    if letter not in letters:
                        continue
                    
                    # Process the image
                    image_path = os.path.join(root, file)
                    features = process_image(image_path)
                    
                    if features is not None:
                        # Save processed features
                        output_path = os.path.join(
                            output_dir,
                            letter,
                            f"sample_{sample_counts[letter]}.npy"
                        )
                        np.save(output_path, features)
                        sample_counts[letter] += 1
        
        # Print statistics
        print("\nProcessed samples per letter:")
        for letter, count in sample_counts.items():
            print(f"{letter}: {count} samples")
        
        print(f"\nTotal processed samples: {sum(sample_counts.values())}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        hands.close()
        
        return output_dir
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None

if __name__ == "__main__":
    processed_data_dir = process_dataset()
    if processed_data_dir:
        print(f"\nProcessed data saved to: {processed_data_dir}")
    else:
        print("\nFailed to process dataset.") 