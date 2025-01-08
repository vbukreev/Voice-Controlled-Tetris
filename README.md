# Voice-Controlled Tetris

The classic game of Tetris but with certain inhancement. This program allows you to play Tetris using voice commands, showcasing the integration of machine learning, audio processing, and game development.

---

## Features

- **Voice Command Integration**:
  - Control Tetris pieces using voice commands: `left`, `right`, `down`, and `rotate`.
- **Real-Time Audio Processing**:
  - Captures audio through the microphone and processes it on-the-fly.
- **Machine Learning Model**:
  - Trained to recognize specific voice commands.
- **Interactive Gameplay**:
  - Fully functional Tetris game implemented with `pygame`.

---

## Technologies Used

- **Python**: Core programming language.
- **PyTorch**: For building and training the voice command recognition model.
- **pyaudio**: To capture live audio input.
- **librosa**: For potential audio data preprocessing and augmentation.
- **pygame**: For Tetris game development and rendering.

---

## How It Works

1. **Audio Input**:
   - The program uses `pyaudio` to capture audio commands in real-time.
2. **Audio Processing**:
   - The audio is normalized and preprocessed to match the input size of the trained model.
3. **Command Prediction**:
   - A neural network model predicts the command (`left`, `right`, `down`, or `rotate`) based on the input audio.
4. **Game Control**:
   - The predicted command is passed to the Tetris game logic, updating the game state accordingly.

---

## Architecture

### Model.py
- **Input**: Raw audio data normalized to a fixed size (44,100 samples for 1 second of audio at 44.1kHz).
- **Layers**: Fully connected neural network with ReLU activations.
- **Output**: One of four possible commands.

### env.py
- **Tetris Logic**: Implemented with grid-based mechanics.
- **Rendering**: Smooth real-time graphics using `pygame`.

---

## Potential Enhancements

- **Improve Voice Recognition Accuracy**:
  - Train the model with a larger and more diverse dataset.
- **Add More Commands**:
  - Commands like `pause` or `restart` for enhanced gameplay control.
- **Optimize Performance**:
  - Use spectrograms and neural networks for better audio classification.

---

## Challenges Addressed

- Real-time audio processing with `pyaudio`.
- Integrating machine learning predictions into an interactive game.
- Creating a smooth gameplay experience with voice control.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Dear Recruiters: Why You Should Care

This project highlights my ability to:
- **Develop End-to-End Applications**: From model training to game development.
- **Solve Real-Time Challenges**: Managing real-time audio processing and gameplay.
- **Think Creatively**: Innovating a classic game with modern AI techniques.

If you're looking for a passionate and skilled developer, you are in the right spot!

