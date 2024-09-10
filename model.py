import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VoiceCommandModel(nn.Module):
    def __init__(self, input_size=44100, num_classes=4):
        """
        Initializes the VoiceCommandModel.

        Args:
        - input_size (int): The size of the input feature vector (e.g., audio sample length).
        - num_classes (int): The number of output classes (e.g., number of commands).
        """
        super(VoiceCommandModel, self).__init__()
        
        # Define the layers of the network
        self.fc1 = nn.Linear(input_size, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 128)         # Second fully connected layer
        self.fc3 = nn.Linear(128, num_classes) # Output layer

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        x = self.fc3(x)          # Output layer
        return x

class VoiceCommandDataset(Dataset):
    def __init__(self, data, labels):
        """
        Initializes the dataset.

        Args:
        - data (np.ndarray): The audio data.
        - labels (np.ndarray): The labels for the audio data.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    """
    Trains the model.

    Args:
    - model (nn.Module): The model to train.
    - train_loader (DataLoader): The data loader for the training data.
    - num_epochs (int): The number of epochs to train for.
    - learning_rate (float): The learning rate for the optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def save_model(model, model_path):
    """
    Saves the model to a file.

    Args:
    - model (nn.Module): The model to save.
    - model_path (str): The path to save the model to.
    """
    torch.save(model.state_dict(), model_path)

def load_model(model_path, input_size=44100, num_classes=4):
    """
    Loads a pre-trained model from a file.

    Args:
    - model_path (str): Path to the model file.
    - input_size (int): The size of the input feature vector.
    - num_classes (int): The number of output classes.

    Returns:
    - VoiceCommandModel: The loaded model.
    """
    model = VoiceCommandModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_audio(audio_data):
    """
    Preprocesses the audio data for the model.

    Args:
    - audio_data (np.ndarray): The raw audio data.

    Returns:
    - np.ndarray: The preprocessed audio data.
    """
    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))
    return audio_data

if __name__ == "__main__":
    # Example usage
    # Create dummy data
    dummy_data = np.random.rand(100, 44100)
    dummy_labels = np.random.randint(0, 4, 100)

    # Create dataset and dataloader
    dataset = VoiceCommandDataset(dummy_data, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize and train the model
    model = VoiceCommandModel()
    train_model(model, train_loader)

    # Save the model
    save_model(model, 'voice_command_model.pth')

    # Load the model
    loaded_model = load_model('voice_command_model.pth')
    print(loaded_model)