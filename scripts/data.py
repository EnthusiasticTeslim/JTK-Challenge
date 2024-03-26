import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def replace_nan(data, replacement_mode='zero', axis=None):
    """
    Replace NaN values in a NumPy array with a specific value, handling all-NaN slices.

    Args:
        data (np.ndarray): The input NumPy array.
        replacement_mode (str): The mode for replacing NaN values. Choices are 'zero', 'mean', or 'median'.
        axis (int or tuple, optional): The axis or axes along which to calculate the mean or median. If None, the mean or median is computed over the entire array.

    Returns:
        np.ndarray: The NumPy array with NaN values replaced, ensuring no all-NaN slice warnings.
    """
    if replacement_mode == 'zero':
        return np.nan_to_num(data, nan=0.0)
    elif replacement_mode in ['mean', 'median']:
        data = np.copy(data)  # Work on a copy of the data to avoid modifying the original array
        
        if replacement_mode == 'mean':
            replacement_value = np.nanmean(data, axis=axis, keepdims=True)
        else:  # 'median'
            replacement_value = np.nanmedian(data, axis=axis, keepdims=True)
        
        # Handle the all-NaN slices by replacing them with a specific value (e.g., 0)
        # This is a simple workaround to avoid the warning and fill those slices
        if np.isnan(replacement_value).any():
            replacement_value[np.isnan(replacement_value)] = 0.0  # You can choose a different fallback value if necessary
        
        np.putmask(data, np.isnan(data), replacement_value)
        return data
    else:
        raise ValueError("Invalid replacement mode. Choose 'zero', 'mean', or 'median'.")




# Define a custom dataset class
class NPZDataset(Dataset):
    def __init__(self, file_paths, mode='zero'):
        self.file_paths = file_paths
        self.mode = mode

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        # Assuming the NPZ files contain arrays named 'array1' and 'array2'
        feature = data['features']
        label = data['label']

        feature = replace_nan(feature, replacement_mode=self.mode, axis=0)
        
        # Convert arrays to PyTorch tensors
        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)
        
        return feature, label