from glob import glob
from multiprocessing import cpu_count
import numpy as np
from pytorch_lightning import LightningDataModule
from random import shuffle
from scipy import stats as st
import torch
from torch.utils.data import Dataset, DataLoader

from utils import normalize_timeseries, replace_nan


class Train_Test_Split:
    def __init__(self, folder, split, cls1_key="AC", cls2_key="PF", val=True):
        self.folder = folder
        self.split = split
        self.label_1 = cls1_key
        self.label_2 = cls2_key
        self.validation = val
    
    def split_data(self):
        val_files = None

        esp_active_file_paths = glob(f"{self.folder}/*_{self.label_1}.npz")
        esp_fail_file_paths = glob(f"{self.folder}/*_{self.label_2}.npz")

        # Shuffle the dataset to prevent repeated training data
        shuffle(esp_active_file_paths)
        shuffle(esp_fail_file_paths)

        # Limit the larger no-failure class to the length of the failure class
        esp_active_file_paths = esp_active_file_paths[:len(esp_fail_file_paths)]

        percent = np.ceil(len(esp_fail_file_paths) * self.split).astype(int)

        # Seperate the training, test and validation files
        train_esp_actv = esp_active_file_paths[:percent]
        train_esp_fail = esp_fail_file_paths[:percent]

        if self.validation != True:
            train_files = train_esp_actv + train_esp_fail
            shuffle(train_files)
        
        else:
            val_percent = -np.ceil(0.1 * len(train_esp_actv)).astype(int)

            train_esp_actv = train_esp_actv[:val_percent]
            train_esp_fail = train_esp_fail[:val_percent]
            train_files = train_esp_actv + train_esp_fail
            shuffle(train_files)

            val_esp_actv = train_esp_actv[val_percent:]
            val_esp_fail = train_esp_fail[val_percent:]
            val_files = val_esp_actv + val_esp_fail
            shuffle(val_files)

        test_esp_actv = esp_active_file_paths[percent:]
        test_esp_fail = esp_fail_file_paths[percent:]
        test_files = test_esp_actv + test_esp_fail
        shuffle(test_files)

        file_path = {"train": train_files, 
                     "val": val_files, 
                     "test": test_files}
        
        return file_path


# Define a custom dataset class
class ESPDataset(Dataset):
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
        label = replace_nan(label, replacement_mode=self.mode, axis=0)
        
        # Convert arrays to PyTorch tensors
        feature = torch.from_numpy(feature).to(torch.float32)
        label = torch.from_numpy(np.array([st.mode(label).mode])).to(torch.float32)

        # Normalize the features
        norm_feats = normalize_timeseries(feature)
        
        return {"features":norm_feats, "labels":label}


# Define a custom dataloader class
class ESPDataModule(LightningDataModule):
    def __init__(self, train_paths, val_paths, test_paths, batch_size):
        super().__init__()
        self.train_data_paths = train_paths
        self.val_data_paths = val_paths
        self.test_data_paths = test_paths
        self.batch_size = batch_size
        self.n_cpus = cpu_count()-2

    def setup(self, stage=None):
        # Create the pytorch datasets
        self.train_dataset = ESPDataset(self.train_data_paths)
        self.val_dataset = ESPDataset(self.val_data_paths)
        self.test_dataset = ESPDataset(self.test_data_paths)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_cpus,
                          persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpus,
                          persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpus,
                          persistent_workers=True)