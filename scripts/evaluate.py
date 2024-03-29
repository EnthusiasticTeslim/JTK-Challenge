import argparse
from data_loader import Train_Test_Split, ESPDataModule
from env import *
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from train import ESPFailureModel


class ESP_Eval_Chkpt:
    def __init__(self, checkpoint_path, batch_size=128):
        self.model = ESPFailureModel.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.batch_size = batch_size
    
    def evaluate(self):
        tts = Train_Test_Split(f"{DAILY_OUTPUT_FOLDER}_{SLIDE_N}", split=0.99)
        data_paths = tts.split_data()
        data_module = ESPDataModule(train_paths=data_paths["train"],
                                    val_paths=data_paths["val"],
                                    test_paths=data_paths["test"],
                                    batch_size=self.batch_size)
        data_module.setup()
        test_dataset = data_module.train_dataset
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=self.batch_size,
                                     shuffle=False,)
        
        trainer = pl.Trainer()
        trainer.test(self.model, dataloaders=test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved model checkpoint.")
    parser.add_argument("--chkpt", type=str, default="checkpoints/best-chckpt-v17.ckpt", help="Model Checkpoint")
    args = parser.parse_args()

    os.system("clear")
    eval = ESP_Eval_Chkpt(args.chkpt)
    eval.evaluate()