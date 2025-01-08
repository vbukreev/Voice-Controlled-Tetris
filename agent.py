import os
import torch
import numpy as np
import pygame
import pyaudio
from model import VoiceCommandModel, load_model
from env import Tetris

class VoiceCommandAgent:
    def __init__(self, model_path, input_size=44100, num_classes=4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = load_model(model_path, input_size, num_classes)
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.record_seconds = 1
        self.pyaudio_instance = pyaudio.PyAudio()

    def get_audio_input(self):
        try:
            stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            frames = []
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk)
                frames.append(np.frombuffer(data, dtype=np.int16))

            stream.stop_stream()
            stream.close()
            return np.hstack(frames)
        except Exception as e:
            print(f"Error capturing audio: {e}")
            return None

    def preprocess_audio(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            return None
        
        
        target_length = 44100
        if len(audio_data) < target_length:
            # Pad with zeros if shorter
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
        elif len(audio_data) > target_length:
            # Trim if longer
            audio_data = audio_data[:target_length]
        
        # Normalize the audio data
        audio_data = audio_data / np.max(np.abs(audio_data))
        return torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

    def predict_action(self, audio_data):
        preprocessed_data = self.preprocess_audio(audio_data)
        if preprocessed_data is None:
            return None
        with torch.no_grad():
            output = self.model(preprocessed_data)
            _, predicted = torch.max(output, 1)
        actions = ['left', 'right', 'down', 'rotate']
        return actions[predicted.item()]

    def run(self):
        pygame.init()
        print("Pygame initialized")
        # Pass the model to the Tetris game
        game = Tetris(self.model)
        clock = pygame.time.Clock()
        running = True

        while running:
            print("Game loop running")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            audio_data = self.get_audio_input()
            action = self.predict_action(audio_data) if audio_data is not None else None
            if action:
                print(f"Action predicted: {action}")
                game.step(action)

            game.draw_board()
            clock.tick(10)

        pygame.quit()
        self.pyaudio_instance.terminate()


if __name__ == "__main__":
    agent = VoiceCommandAgent("voice_command_model.pth")
    agent.run()
