import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from env import *#get_n_value
from utils import date_parser

os.system("clear")
n_val = get_n_value(os.getcwd())

training_folder_path = "Train" # Folder path


class JTK_Preprocess_ESP:
    def __init__(self, path, slide) -> None:
        self.data_path = path
        self.esp_tracker = None
        self.wells = None
        self.n = slide
    
    def load_esp(self):
        """Import the Esp tracker file"""
        esp_df = pd.read_excel(f"{self.data_path}/esp_tracker_train.xlsx",
                            index_col=0,
                            date_parser=date_parser,
                            parse_dates=["Ins Date","Startup Date","Pull date","Corrected Failure Date"])
        esp_df.columns = [name.replace(" ","_") for name in esp_df.columns]
        esp_df = esp_df.reset_index(drop=True)
        esp_df["Label_Start_Time"] = esp_df.apply(lambda x: x["Corrected_Failure_Date"] - pd.Timedelta(f"{self.n} day"), axis=1)
        self.esp_tracker = esp_df
        return esp_df
    
    def create_well_paths(self):
        """Extract unique wells and create a file path list"""
        esp_df = self.esp_tracker
        well_api = sorted(esp_df["API"].unique().tolist())
        csv_paths = [f"{self.data_path}/{api}.csv" for api in well_api]
        self.wells = well_api
        return csv_paths
    
    def crop_testdata_esplabels(self, clean=True):
        """
        Crop the well data for each installed pump
        
        Args:
            clean (bool): Delete the preprocessing folder if it exists. Defaults to True.
        """
        fol_name = f"Processed_{self.n}"
        if clean and os.path.exists(fol_name):
            os.system(f"rm -rf {fol_name}")
        
        if not os.path.exists(fol_name):
            os.makedirs(fol_name)
        
        espt = self.load_esp()
        pump_data_path = self.create_well_paths()

        for idx,well in tqdm(enumerate(self.wells), position=0, total=len(self.wells)):
            # Load the well test tracker for each unique well
            well_tests = espt[espt["API"]==well].sort_values("Install_#").reset_index(drop=True)

            # Import the csv file for each well. Remove timezone since it is UTC +00:00
            well_df = pd.read_csv(pump_data_path[idx],index_col=0)
            well_df.index = pd.to_datetime(well_df.index,format="mixed").tz_localize(None)
            well_df = well_df.sort_index()

            for tdx,row in well_tests.iterrows():
                pump_num = row["Install_#"]
                install_dt = row["Ins_Date"]
                pull_dt = row["Pull_date"]
                fail_dt = row["Corrected_Failure_Date"]

                # Crop the timeseries to the install and pull date of the test
                if pull_dt is not np.nan:
                    pump_crop = well_df.loc[install_dt:pull_dt,:]
                else:
                    pump_crop = well_df.loc[install_dt:,:]
                
                if len(pump_crop)>0:
                    # Generate classification timeseries labels. Only include a label if the test failed
                    pump_crop.loc[:,["Label"]] = np.zeros(len(pump_crop))
                    if fail_dt is not np.nan:
                        pump_crop.loc[fail_dt:,["Label"]] = 1
                    pump_crop.loc[:,["Label"]] = pump_crop["Label"].astype(int)

                    if not os.path.exists(f"{fol_name}/{well}"):
                        os.makedirs(f"{fol_name}/{well}")
                        pump_crop.to_parquet(f"{fol_name}/{well}/esp_test_{pump_num}.parquet")
                else:
                    pass
        return


if __name__ == "__main__":
    #################### Call the function ####################
    prep = JTK_Preprocess_ESP(path=training_folder_path, slide=n_val)
    # esp_df = prep.load_esp() # This line is not needed except for QC
    prep.crop_testdata_esplabels()
    os.system("rm -rf scripts/__pycache__")