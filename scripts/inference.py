import argparse
from env import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from train import ESPFailureModel
import torch
from utils import resample_and_interpolate_features as rsif, \
    replace_nan, normalize_timeseries

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cropped_14/z8jfoj1ef3_esp#2_2022-08-03_PF.npz
class ESP_Predictor:
    def __init__(self, checkpoint_path, api, csv_folder_path="Train", probability=0.85):
        self.model = ESPFailureModel.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.api = api
        self.main_folder = csv_folder_path
        self.probability = probability
        self.dates = []
        self.well = None
        self.data = None
    
    def load_well_api(self):
        # Import the csv file for each well. Remove timezone since it is UTC +00:00
        well_df = pd.read_csv(f"{self.main_folder}/{self.api}.csv",index_col=0)
        well_df.index = pd.to_datetime(well_df.index,format="mixed").tz_localize(None)
        well_df = well_df.sort_index()
        self.well = well_df
    
    def preprocess_files(self):
        df = self.well
        data = []

        for data_date in np.unique(df.index.date):
            ######### Interpolate
            daily_df = df[df.index.date == data_date]
            # Dummy label column to make it compatible with original function usecase
            daily_df.loc[:,["Label"]] = None
            rs_daily_df = rsif(data_date, RESAMPLING_FREQ, daily_df)
            # Drop dummy column
            rs_daily_df.drop(["Label"], axis=1, inplace=True)

            ######### Fill nans and normalize tensor
            arr = rs_daily_df.to_numpy()
            arr = replace_nan(arr, replacement_mode="zero", axis=0)
            tensor = feature = torch.from_numpy(arr).to(torch.float32)
            tensor = normalize_timeseries(tensor)
            data.append(tensor)
            self.dates.append(data_date)
        
        self.data = torch.stack(data)

    def predict(self, save_csv=True):
        self.load_well_api()
        self.preprocess_files()
        pred_probs = self.model(self.data.to(DEVICE))
        pred_probs = np.squeeze(pred_probs[1].cpu().detach().numpy())
        class_label = np.where(pred_probs >= self.probability, 1, 0)

        predictions = pd.DataFrame({"API": self.api,
                                    "Date": self.dates,
                                    "Fail": class_label,
                                    "Probability": pred_probs})
        
        if save_csv:
            folder = f"{PRED_FOL}_{SLIDE_N}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            predictions.to_csv(f"{folder}/pred_{self.api}.csv", index=False)
        
        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved model checkpoint.")
    parser.add_argument("--chkpt", type=str, default=BEST_CHECKPOINT, help="Model Checkpoint")
    parser.add_argument("--api", type=str, help="Well api number")
    parser.add_argument("--train_folder", default="Train", type=str, help="Training data folder path")
    parser.add_argument("--prob", type=float, default=PROBA_THRESHOLD, help="Prediction probability threshold")
    args = parser.parse_args()

    args.api = "z8jfojo31x"

    os.system("clear")

    pred = ESP_Predictor(checkpoint_path=args.chkpt,
                         api=args.api,
                         csv_folder_path=args.train_folder,
                         probability=args.prob)
    pred.predict()

    os.system("rm -rf scripts/__pycache__")