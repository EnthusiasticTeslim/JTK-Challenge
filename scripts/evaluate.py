import argparse
from data_loader import Train_Test_Split, ESPDataModule
from env import *
import numpy as np
import os
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from train import ESPFailureModel

from env import PROBA_THRESHOLD


class ESP_Eval_Chkpt:
    def __init__(self, checkpoint_path, batch_size=128):
        self.model = ESPFailureModel.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.batch_size = batch_size
        self.dataloader = None
    
    def prep_dataloader(self):
        tts = Train_Test_Split(f"{DAILY_OUTPUT_FOLDER}_{SLIDE_N}", split=0.90)
        data_paths = tts.split_data()
        data_module = ESPDataModule(train_paths=data_paths["train"],
                                    val_paths=data_paths["val"],
                                    test_paths=data_paths["test"],
                                    batch_size=self.batch_size)
        data_module.setup()
        test_dataset = data_module.test_dataset
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=self.batch_size,
                                     shuffle=False,)
        self.dataloader = test_dataloader
    
    def evaluate(self):
        trainer = pl.Trainer()
        trainer.test(self.model, dataloaders=self.dataloader)

    def confusion_matrix(self):
        ytest = []
        ypred = []
        for batch in self.dataloader:
            yp = self.model(batch["features"])
            yp = np.squeeze(yp[1].cpu().detach().numpy())
            yp = np.where(yp>=PROBA_THRESHOLD, 1, 0)
            yt = np.squeeze(batch["labels"].cpu().detach().numpy())
            ytest.append(yt)
            ypred.append(yp)
        ytest = np.concatenate(ytest)
        ypred = np.concatenate(ypred)
        cm = confusion_matrix(ytest, ypred)
        print("──────────────────",
              "Confusion Matrix",
              "──────────────────",
              cm,
              "──────────────────", sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved model checkpoint.")
    parser.add_argument("--chkpt", type=str, default=BEST_CHECKPOINT, help="Model Checkpoint")
    args = parser.parse_args()

    os.system("clear")
    
    eval = ESP_Eval_Chkpt(args.chkpt)
    eval.prep_dataloader()
    eval.evaluate()
    eval.confusion_matrix()

    os.system("rm -rf scripts/__pycache__")