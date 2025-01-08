import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VoiceCommandModel(nn.Module):
    def __init__(self, input_size=44100, num_classes=4):
    
        super(VoiceCommandModel, self).__init__()
        
        # layers of the network
        self.fc1 = nn.Linear(input_size, 256)  # connected layer
        self.fc2 = nn.Linear(256, 128)         # fully connected layer
        self.fc3 = nn.Linear(128, num_classes) # Output layer

    def forward(self, x):
        
        x = F.relu(self.fc1(x))  # ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # ReLU activation to the second layer
        x = self.fc3(x)          # Output layer
        return x

class VoiceCommandDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    
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
    
    torch.save(model.state_dict(), model_path)

def load_model(model_path, input_size=44100, num_classes=4):
    
    model = VoiceCommandModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_audio(audio_data):
    
    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))
    return audio_data

if __name__ == "__main__":
    # Create data
    data = np.random.rand(100, 44100)
    labels = np.random.randint(0, 4, 100)

    # Create dataset and dataloader
    dataset = VoiceCommandDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize and train the model
    model = VoiceCommandModel()
    train_model(model, train_loader)

 
    save_model(model, 'voice_command_model.pth')

  
    loaded_model = load_model('voice_command_model.pth')
    print(loaded_model)