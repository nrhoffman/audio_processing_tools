import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import yaml

from early_stopping import EarlyStopping
from audio_dataset import AudioDataset
from crnn_model import CRNNModel

yaml_path = "./models/data.yaml"

with open(yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)

train_path = data_yaml['train']
val_path = data_yaml['val']
test_path = data_yaml['test']


train_dataset = AudioDataset(csv_file=os.path.join(train_path, 'train_pairs.csv'), root_dir=train_path)
val_dataset = AudioDataset(csv_file=os.path.join(val_path, 'val_pairs.csv'), root_dir=val_path)
test_dataset = AudioDataset(csv_file=os.path.join(test_path, 'test_pairs.csv'), root_dir=test_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CRNNModel(n_mels=128, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

early_stopping = EarlyStopping(patience=5, verbose=True)

best_val_loss = float('inf')

for epoch in range(10):
    model.train()  # Set model to training mode
    running_loss = 0.0
    total_batches = len(train_loader)

    # Training phase
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU if available

        optimizer.zero_grad()  # Zero the gradients before each batch

        # Forward pass
        outputs = model(X_batch)
        y_batch = model.transform_batch(y_batch)

        loss = criterion(outputs, y_batch)  # Compute loss

        # Backward pass
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model parameters

        running_loss += loss.item() * X_batch.size(0)  # Accumulate loss
        progress = (batch_idx + 1) / total_batches * 100
        print(f'\rTraining... Epoch [{epoch+1}/20], Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}, Progress: {progress:.2f}%', end='')

    epoch_loss = running_loss / len(train_loader.dataset)  # Average loss per epoch
    print(f'Epoch [{epoch+1}/10], Loss: {epoch_loss:.4f}')

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    total_val_batches = len(val_loader)

    with torch.no_grad():  # Disable gradient calculation for validation
        for val_batch_idx, (X_val_batch, y_val_batch) in enumerate(val_loader):
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            # Forward pass
            val_outputs = model(X_val_batch)
            y_val_batch = model.transform_batch(y_val_batch)

            loss = criterion(val_outputs, y_val_batch)

            val_loss += loss.item() * X_val_batch.size(0)  # Accumulate validation loss

            progress = (val_batch_idx + 1) / total_val_batches * 100
            print(f'\rValidating... Epoch [{epoch+1}/20], Batch [{val_batch_idx+1}/{total_val_batches}], Validation Loss: {loss.item():.4f}, Progress: {progress:.2f}%', end='')

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    early_stopping(avg_val_loss)
    if early_stopping.should_stop:
        print("Early stopping triggered!")
        break

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        # Save the model state
        torch.save(model.state_dict(), f'best_model.pth')

    torch.save(model.state_dict(), 'latest_model.pth')

train_loader = None
val_loader = None

test_loss = 0.0

# Initialize lists to store predictions and actual values for further analysis (optional)
all_predictions = []
all_actuals = []

with torch.no_grad():  # Disable gradient calculation for testing
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)

        # Forward pass
        test_outputs = model(X_test_batch)
        y_test_batch = model.transform_batch(y_test_batch)

        loss = criterion(test_outputs, y_test_batch)

        test_loss += loss.item() * X_test_batch.size(0)  # Accumulate test loss
        
        # Store predictions and actual values
        all_predictions.append(test_outputs.cpu().numpy())  # Move to CPU and convert to numpy
        all_actuals.append(y_test_batch.cpu().numpy())

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader.dataset)
print(f'Test Loss: {avg_test_loss:.4f}')

# If you're interested in analyzing predictions, you can concatenate the lists
all_predictions = np.concatenate(all_predictions)
all_actuals = np.concatenate(all_actuals)

num_samples_to_display = 5
for i in range(num_samples_to_display):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Predicted Spectrogram")
    plt.imshow(all_predictions[i].T, aspect='auto', origin='lower')  # Transpose if necessary
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Actual Spectrogram")
    plt.imshow(all_actuals[i].T, aspect='auto', origin='lower')  # Transpose if necessary
    plt.colorbar()

    plt.show()