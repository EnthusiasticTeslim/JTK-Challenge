from env import *
from inference import ESP_Predictor
import os
import pandas as pd
from tqdm import tqdm
from utils import date_parser

from env import PROBA_THRESHOLD

if __name__ == "__main__":
    os.system("clear")

    # Load the dataframe
    data_path = "Train"
    esp_tracker = pd.read_excel(f"{data_path}/esp_tracker_train.xlsx",
                                index_col=0,
                                date_parser=date_parser,
                                parse_dates=["Ins Date","Startup Date","Pull date","Corrected Failure Date"])
    esp_tracker.columns = [name.replace(" ","_") for name in esp_tracker.columns]
    esp_tracker = esp_tracker.reset_index(drop=True)

    all_apis = esp_tracker.API.unique().tolist()

    for api in tqdm(all_apis, total=len(all_apis), position=0):
        model = ESP_Predictor(checkpoint_path=BEST_CHECKPOINT, 
                              api=api, 
                              csv_folder_path=data_path, 
                              probability=PROBA_THRESHOLD)
        pred = model.predict()

    os.system("rm -rf scripts/__pycache__")