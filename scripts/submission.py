import argparse
from env import *
from inference import ESP_Predictor
import os
import pandas as pd
from tqdm import tqdm
from utils import date_parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate all well(api) data.")
    parser.add_argument("--folder", default="Train", type=str, help=" data folder path")
    args = parser.parse_args()

    os.system("clear")

    # Load the dataframe
    esp_tracker = pd.read_excel(f"{args.folder}/esp_tracker_train.xlsx",
                                index_col=0,
                                date_parser=date_parser,
                                parse_dates=["Ins Date","Startup Date","Pull date","Corrected Failure Date"])
    esp_tracker.columns = [name.replace(" ","_") for name in esp_tracker.columns]
    esp_tracker = esp_tracker.reset_index(drop=True)

    all_apis = esp_tracker.API.unique().tolist()

    for api in tqdm(all_apis, total=len(all_apis), position=0):
        model = ESP_Predictor(checkpoint_path=BEST_CHECKPOINT, 
                              api=api, 
                              csv_folder_path=args.folder, 
                              probability=0.8)
        pred = model.predict()

    os.system("rm -rf scripts/__pycache__")