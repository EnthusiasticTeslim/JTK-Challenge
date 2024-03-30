import argparse
from data_loader import Train_Test_Split, ESPDataset, ESPDataModule
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from env import *
from train import ESPFailureModel
from data_loader import Train_Test_Split, ESPDataset, ESPDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger



def objective(trial, seed, split):
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    num_stack_layers = trial.suggest_int('num_stack_layers', 1, 3)
    num_epochs = trial.suggest_categorical('num_epochs', [150, 200, 250, 300])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    # set seed for reproducibility
    pl.seed_everything(seed=seed)

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
                            hidden_size=hidden_size,
                            num_stack_layers=num_stack_layers,
                            n_layers=num_layers)

    # Define the model callbacks
    checkpoint_call_back = ModelCheckpoint(
        dirpath=f"checkpoints/{trial.number}",
        filename="best-chckpt",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    
    logger = TensorBoardLogger(save_dir="lightning_logs", name="JTK_Challenge")

    trainer = pl.Trainer(logger=logger,
                         callbacks=checkpoint_call_back,
                         max_epochs=num_epochs,
                         deterministic=True,
                         enable_progress_bar=True)
    
    model.save_hyperparameters({"hidden_dim": hidden_size,
                            "learning_rate": learning_rate,
                            "dropout": dropout,
                            "num_stack_layers": num_stack_layers,
                            "num_layers": num_layers,
                            "num_epochs": num_epochs,
                            "seed": seed,
                            "split": split,
                            "batch_size": batch_size,
                            "device":trainer.accelerator})
    
    trainer.fit(model, data_module)
    
    return trainer.callback_metrics["val_loss"].item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for ESP Failure model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--n_trials", type=int, default=300)
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args.seed, args.split), n_trials=args.n_trials)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    print("Best value:", study.best_value)