import argparse
from data_loader import Train_Test_Split, ESPDataset, ESPDataModule
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import LSTMClassifier
from torch import nn, optim
from torchmetrics.classification import BinaryAccuracy
import torch 
from torchmetrics.classification import BinaryFBetaScore

from env import *


class ESPFailureModel(pl.LightningModule):
    def __init__(self, n_features, n_classes, lr, dropout, n_layers):
        super().__init__()
        self.model = LSTMClassifier(n_features=n_features, 
                                    n_classes=n_classes,
                                    n_layer=n_layers,
                                    dropout=dropout)
        self.criterion = nn.BCELoss()
        self.lr = lr
        self.metric = BinaryAccuracy(threshold=0.5)
        self.fbeta_score = BinaryFBetaScore(beta=2.0)

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0

        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        daily_sequence = batch["features"]
        labels = batch["labels"]
        loss, outputs = self(daily_sequence, labels)
        predictions = torch.round(outputs)
        step_acc = self.metric(predictions, labels)
        step_fbeta = self.fbeta_score(predictions, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", step_acc, prog_bar=True, logger=True)
        self.log("train_fbeta", step_fbeta, prog_bar=True, logger=True)
        return {"loss": loss, "acc": step_acc, "fbeta": step_fbeta}
    
    def validation_step(self, batch, batch_idx):
        daily_sequence = batch["features"]
        labels = batch["labels"]
        loss, outputs = self(daily_sequence, labels)
        predictions = torch.round(outputs)
        step_acc = self.metric(predictions, labels)
        step_fbeta = self.fbeta_score(predictions, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", step_acc, prog_bar=True, logger=True)
        self.log("val_fbeta", step_fbeta, prog_bar=True, logger=True)
        return {"loss": loss, "acc": step_acc, "fbeta": step_fbeta}
    
    def test_step(self, batch, batch_idx):
        daily_sequence = batch["features"]
        labels = batch["labels"]
        loss, outputs = self(daily_sequence, labels)
        predictions = torch.round(outputs)
        step_acc = self.metric(predictions, labels)
        step_fbeta = self.fbeta_score(predictions, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", step_acc, prog_bar=True, logger=True)
        self.log("test_fbeta", step_fbeta, prog_bar=True, logger=True)
        return {"loss": loss, "acc": step_acc, "fbeta": step_fbeta}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(),  lr=self.lr)



def trainer_wrapper(split, batch_size, learning_rate, num_epochs, dropout, num_layers):
    # ------------------------------ Train test split and load dataset ------------------------------
    tts = Train_Test_Split(f"{DAILY_OUTPUT_FOLDER}_{SLIDE_N}", split=split)
    data_paths = tts.split_data()

    # Create the dataloaders
    data_module = ESPDataModule(train_paths=data_paths["train"],
                                val_paths=data_paths["val"],
                                test_paths=data_paths["test"],
                                batch_size=batch_size)

    # Load a single file to get the model dimensions
    single_batch = next(iter(ESPDataset(data_paths["val"][:1])))
    n_features = single_batch["features"].shape[-1]
    n_classes= single_batch["labels"].shape[-1]

    # Initialize the model
    model = ESPFailureModel(n_features=n_features,
                            n_classes=n_classes,
                            lr=learning_rate,
                            dropout=dropout,
                            n_layers=num_layers)

    # Define the model callbacks
    checkpoint_call_back = ModelCheckpoint(dirpath="checkpoints",
                                           filename="best-chckpt",
                                           save_top_k=1,
                                           verbose=True,
                                           monitor="val_loss",
                                           mode="min")
    
    logger = TensorBoardLogger(save_dir="lightning_logs", name="JTK_Challenge")

    trainer = pl.Trainer(logger=logger,
                         callbacks=checkpoint_call_back,
                         max_epochs=num_epochs,
                         enable_progress_bar=True)
    
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM model on NPZ data with dynamic configuration.")
    parser.add_argument("--split", type=float, default=0.9, help="Train test split percentage")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the DataLoader")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    # parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of the LSTM hidden state")
    # parser.add_argument("--step_size", type=int, default=5, help="Period of learning rate decay")
    # parser.add_argument("--gamma", type=float, default=0.1, help="Multiplicative factor of learning rate decay")
    # parser.add_argument("--bidirectional", action="store_true", help="Enable bidirectional LSTM")

    args = parser.parse_args()

    os.system("clear")

    trainer_wrapper(split=args.split,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate, 
                    num_epochs=args.num_epochs,
                    dropout=args.dropout,
                    num_layers=args.num_layers)
    
    os.system("rm -rf scripts/__pycache__")