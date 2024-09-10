import pygame
import random
import numpy as np
import torch
import pyaudio
import wave
from model import VoiceCommandModel, load_model

# Initialize pygame
pygame.init()

# Define constants
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [
    (0, 255, 255),  # I
    (255, 0, 255),  # T
    (0, 255, 0),    # S
    (255, 0, 0),    # Z
    (255, 255, 0),  # O
    (255, 165, 0),  # L
    (0, 0, 255)     # J
]

# Define shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]],  # Z
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]]   # J
]

class Tetris:
    def __init__(self, model):
        self.board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.current_piece = None
        self.current_color = None
        self.current_position = (0, 0)
        self.spawn_piece()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris')
        self.model = model

    def spawn_piece(self):
        shape_index = random.randint(0, len(SHAPES) - 1)
        self.current_piece = SHAPES[shape_index]
        self.current_color = COLORS[shape_index]
        self.current_position = (0, GRID_WIDTH // 2 - len(self.current_piece[0]) // 2)

    def rotate_piece(self):
        self.current_piece = np.rot90(self.current_piece)

    def move_piece(self, dx, dy):
        new_x = self.current_position[0] + dx
        new_y = self.current_position[1] + dy
        if self.is_valid_position(new_x, new_y):
            self.current_position = (new_x, new_y)

    def is_valid_position(self, x, y):
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    if x + i >= GRID_HEIGHT or y + j < 0 or y + j >= GRID_WIDTH:
                        return False
                    if self.board[x + i][y + j]:
                        return False
        return True

    def lock_piece(self):
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    self.board[self.current_position[0] + i][self.current_position[1] + j] = 1
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        new_board = [row for row in self.board if not all(row)]
        lines_cleared = GRID_HEIGHT - len(new_board)
        self.board = np.vstack([np.zeros((lines_cleared, GRID_WIDTH), dtype=int), new_board])

    def is_game_over(self):
        return not self.is_valid_position(*self.current_position)

    def step(self, action):
        if action == 'left':
            self.move_piece(0, -1)
        elif action == 'right':
            self.move_piece(0, 1)
        elif action == 'down':
            self.move_piece(1, 0)
        elif action == 'rotate':
            self.rotate_piece()

        if not self.is_valid_position(self.current_position[0] + 1, self.current_position[1]):
            self.lock_piece()
            if self.is_game_over():
                return 'game_over'

        return 'continue'

    def draw_board(self):
        self.screen.fill(BLACK)
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                if self.board[i][j]:
                    pygame.draw.rect(self.screen, WHITE, (j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.current_color, (
                        (self.current_position[1] + j) * BLOCK_SIZE,
                        (self.current_position[0] + i) * BLOCK_SIZE,
                        BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.update()

    def get_audio_input(self):
        # Record audio for a short duration and return it as a numpy array
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        record_seconds = 1

        p = pyaudio.PyAudio()
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []

        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.hstack(frames)
        return audio_data

    def predict_action(self, audio_data):
        # Preprocess audio data and predict action
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        audio_data = audio_data.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(audio_data)
            _, predicted = torch.max(output, 1)
        
        actions = ['left', 'right', 'down', 'rotate']
        return actions[predicted.item()]

def main():
    pass #load