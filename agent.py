import torch
import numpy as np
import pygame
import pyaudio
from model import VoiceCommandModel, load_model
from env import Tetris

class VoiceCommandAgent:
    def __init__(self, model_path, input_size=44100, num_classes=4):
        """
        Initializes the VoiceCommandAgent.

        Args:
        - model_path (str): Path to the trained model file.
        - input_size (int): The size of the input feature vector.
        - num_classes (int): The number of output classes.
        """
        self.model = load_model(model_path, input_size, num_classes)
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.record_seconds = 1
        self.pyaudio_instance = pyaudio.PyAudio()

    def get_audio_input(self):
        """
        Captures audio input from the microphone.

        Returns:
        - np.ndarray: The captured audio data.
        """
        stream = self.pyaudio_instance.open(format=self.audio_format,
                                            channels=self.channels,
                                            rate=self.rate,
                                            input=True,
                                            frames_per_buffer=self.chunk)

        frames = []

        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

        audio_data = np.hstack(frames)
        return audio_data

    def preprocess_audio(self, audio_data):
        """
        Preprocesses the audio data for the model.

        Args:
        - audio_data (np.ndarray): The raw audio data.

        Returns:
        - torch.Tensor: The preprocessed audio data as a tensor.
        """
        # Normalize the audio data
        audio_data = audio_data / np.max(np.abs(audio_data))
        return torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

    def predict_action(self, audio_data):
        """
        Predicts the action from the audio data using the model.

        Args:
        - audio_data (np.ndarray): The raw audio data.

        Returns:
        - str: The predicted action.
        """
        preprocessed_data = self.preprocess_audio(audio_data)
        with torch.no_grad():
            output = self.model(preprocessed_data)
            _, predicted = torch.max(output, 1)
        
        actions = ['left', 'right', 'down', 'rotate']
        return actions[predicted.item()]

    def run(self):
        """
        Runs the agent to control the Tetris game using voice commands.
        """
        game = Tetris(self.model)
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get audio input and predict action
            audio_data = self.get_audio_input()
            action = self.predict_action(audio_data)

            # Execute the predicted action
            game.step(action)

            game.draw_board()
            clock.tick(10)

        pygame.quit()

if __name__ == "__main__":
    # Example usage
    agent = VoiceCommandAgent('path_to_your_trained_model.pth')
    agent.run()