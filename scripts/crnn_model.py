import torch.nn as nn
import torch

class CRNNModel(nn.Module):
    def __init__(self, n_mels=128, num_classes=2):
        super(CRNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d((2, 2))
        
        # Recurrent layer (LSTM)
        self.lstm = nn.LSTM(input_size=n_mels // 2, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(128 * 2, 64)  # Bidirectional LSTM, hence 128 * 2
        
    def forward(self, x):
        x = self.transform_batch(x)

        # Recurrent layers
        x, _ = self.lstm(x)
        
        # Classification layer
        x = self.fc(x)  # Take the output at the last time step
        
        return x
    
    def transform_batch(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Same conv1 and pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Same conv2 and pooling
        
        # Reshape y_batch for LSTM (batch_size, time_frames, n_mels//2)
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, height * width, channels)

        return x