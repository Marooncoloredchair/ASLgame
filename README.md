# ASL Sign Language Recognition Web App

A modern, real-time American Sign Language (ASL) letter recognition game and web application powered by deep learning and computer vision.

## Overview
This project uses a webcam and a trained neural network to recognize ASL letters in real time. Users play a word game by signing each letter of a word, with instant feedback and scoring. The app is built with Python, Flask, TensorFlow/Keras, and MediaPipe.

## Features
- Real-time ASL letter recognition using your webcam
- Interactive word game with categories (Animals, Food, Colors, Simple Words)
- Visual feedback, progress bar, and scoring
- Robust backend with smoothed prediction logic for reliability
- Clean, responsive web interface

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ASLgame.git
   cd ASLgame
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download or train a model:**
   - Place your trained model file (e.g., `new_asl_model.h5`) in the project root.
   - (Optional) Use the provided scripts to preprocess data and train your own model.
4. **Run the web app:**
   ```bash
   python web_game.py
   ```
5. **Open your browser:**
   - Go to `http://localhost:5000` to play the game!

## Usage
- Show your hand sign clearly in the camera.
- Hold the sign steady until it's detected.
- Complete the word by signing each letter in sequence.
- Get points for each completed word.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
