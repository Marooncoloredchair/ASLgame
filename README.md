# ASL Letter Game

An interactive American Sign Language (ASL) learning game that uses computer vision to detect hand signs and help users practice ASL letters.

## Features

- Real-time ASL letter detection using MediaPipe and TensorFlow
- Multiple word categories (Animals, Food, Colors, Simple Words)
- Score tracking
- Visual feedback and hand landmark display
- Confidence-based letter recognition

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- TensorFlow
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Marooncoloredchair/ASLgame.git
cd ASLgame
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have a trained model file (`new_asl_model.h5`) in the project directory
2. Run the game:
```bash
python asl_game.py
```

3. Game Controls:
- Show hand signs to the camera to play
- Press 'q' to quit the game

## Game Rules

1. The game will show you a word from a random category
2. Sign each letter of the word in sequence
3. Get points for completing words correctly
4. The game will automatically move to the next word when you complete the current one

## Categories

- Animals: CAT, DOG, BIRD, FISH, BEAR, LION, TIGER, EAGLE
- Food: PIZZA, BURGER, SALAD, APPLE, BANANA, GRAPE, BREAD, CAKE
- Colors: RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, BLACK, WHITE
- Simple Words: HELLO, LOVE, PEACE, HAPPY, SMILE, DREAM, HOPE, JOY

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
