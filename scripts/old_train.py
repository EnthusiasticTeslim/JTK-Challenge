import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
import argparse
from datetime import datetime
from pathlib import Path
import glob
from scripts.data_loader import NPZDataset
from torch.optim.lr_scheduler import StepLR


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim: int, num_layers: int, bidirectional: bool, dropout: float):
        super(LSTMClassifier, self).__init__()
        # Ensure dropout is set to 0 if num_layers is 1 to avoid runtime error
        dropout = 0 if num_layers == 1 else dropout
        self.input_norm = nn.BatchNorm1d(input_dim)
        #self.bn_input = nn.BatchNorm1d(input_dim) 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim * (2 if bidirectional else 1))
        #self.bn_output = nn.BatchNorm1d(hidden_dim * (2 if bidirectional else 1)) 
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Since the data is already padded, directly pass x to LSTM
        x = x.float()
        #x = self.bn_input(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        # Apply BN to outputs; permute as needed for BatchNorm
        #lstm_out = self.bn_output(lstm_output.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Take the last timestep
        #if self.lstm.bidirectional:
        #    lstm_out = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size], lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        #else:
        #    lstm_out = lstm_out[:, -1, :]

        lstm_out = self.bn(lstm_out[:, -1, :])
        
        out = self.fc(lstm_out)
        return out

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, 
          criterion: nn.Module, num_epochs: int, device: torch.device, log_interval: int = 10,
          checkpoint_path: str = "best_model.pt") -> None:
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # check for NaN values in the input data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #for name, param in model.named_parameters():
            #    if param.requires_grad:
            #        print(name, param.data)
            #    if torch.isnan(param.data).any():
            #        print('output has NaN values')
            #        break
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | Train Loss: {train_loss/(batch_idx+1):.4f}')

        #scheduler.step() 

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch: {epoch+1}/{num_epochs} | Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'New best model saved at {checkpoint_path}')

    print(f'Training completed. Best validation loss: {best_val_loss:.4f}')


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the LSTM classifier model on the test set.

    Args:
        model (nn.Module): The LSTM classifier model.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for evaluation (CPU or GPU).

    Returns:
        Tuple[float, float]: Test loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) 
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy


def main(directory, batch_size, learning_rate, num_layers, num_epochs,
         hidden_dim, step_size, gamma, dropout, bidirectional):
   
    root = Path(__file__).resolve().parents[1]
    directory_path = root / directory
    
    # Construct the glob pattern to match all .npz files in the directory
    npz_pattern = str(directory_path / '*.npz')
    
    # Find all matching .npz files
    file_paths = glob.glob(npz_pattern)
    
    print(f"Found {len(file_paths)} .npz files in the directory '{directory_path}'")

    # Sort and split the file paths as before
    file_paths.sort(key=lambda x: datetime.strptime(x.split('_')[-1].split('.')[0], '%Y-%m-%d'))
    num_files = len(file_paths)
    train_split = int(0.6 * num_files)
    val_split = int(0.8 * num_files)
    train_paths = file_paths[:train_split]
    val_paths = file_paths[train_split:val_split]
    test_paths = file_paths[val_split:]

    # Create dataset instances and dataloaders
    train_dataset = NPZDataset(train_paths)
    val_dataset = NPZDataset(val_paths)
    test_dataset = NPZDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    single_batch = next(iter(train_loader))
    input_batch = single_batch[0].numpy()
    target_batch = single_batch[1].numpy()

    input_dim = input_batch[0].shape #train_loader.dataset[0][0
    output_dim = target_batch[0].shape
    
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    #print(f"Input dimension: {train_loader[0][0].shape}, Output dimension: {train_loader[0][1].shape}")


    # Initialize the model
    model = LSTMClassifier(input_dim=13,
                           output_dim=output_dim,   
                           hidden_dim=96,
                           num_layers=num_layers,
                           bidirectional=bidirectional, 
                           dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train the model
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, device)

    # Evaluate the model
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM model on NPZ data with dynamic configuration.")
    parser.add_argument('--directory', type=str, required=True, help='Directory containing the NPZ files')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the DataLoader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the LSTM hidden state')
    parser.add_argument('--step_size', type=int, default=5, help='Period of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Enable bidirectional LSTM')

    args = parser.parse_args()
    main(args.directory, args.batch_size, args.learning_rate, args.num_layers, args.num_epochs,
         args.hidden_dim, args.step_size, args.gamma, args.dropout, args.bidirectional)

