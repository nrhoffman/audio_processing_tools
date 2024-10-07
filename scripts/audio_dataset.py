import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        df = pd.read_csv(csv_file)
        df= df[~df['Mixed'].isin(['Mixed']) & ~df['Clean'].isin(['Clean'])]
        self.df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the file paths
        mixed_file = self.df.iloc[idx]['Mixed']
        clean_file = self.df.iloc[idx]['Clean']

        # Load audio files for this sample
        mixed_np = np.load(os.path.join(self.root_dir, mixed_file))
        clean_np = np.load(os.path.join(self.root_dir, clean_file))
        
        # Expand dimensions to match model input shape
        mixed_np = np.expand_dims(mixed_np, axis=0)  # Add channel dimension
        clean_np = np.expand_dims(clean_np, axis=0)
        
        # Convert to torch tensors
        mixed_tensor = torch.tensor(mixed_np, dtype=torch.float32)
        clean_tensor = torch.tensor(clean_np, dtype=torch.float32)

        mixed_tensor = self.pad_tensor(mixed_tensor, 376)
        clean_tensor = self.pad_tensor(clean_tensor, 376)

        return mixed_tensor, clean_tensor

    def pad_tensor(self, tensor, target_size):
        """
        Pads the tensor along the last dimension to the target size.
        """
        current_size = tensor.size(-1)
        if current_size < target_size:
            pad_size = target_size - current_size
            tensor = F.pad(tensor, (0, pad_size))  # Pad along the last dimension
        return tensor