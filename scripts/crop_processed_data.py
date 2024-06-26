import argparse
from glob import glob
import os
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from scipy import stats as st

# Import environment variables
from env import SLIDE_N,\
    ESP_OUTPUT_FOLDER,\
    DAILY_OUTPUT_FOLDER,\
    RESAMPLING_FREQ
from utils import clip_zscore_outliers, resample_and_interpolate_features

os.system("clear")

# Reassign the environment variable
input_path = f"{ESP_OUTPUT_FOLDER}_{SLIDE_N}"
output_path = f"{DAILY_OUTPUT_FOLDER}_{SLIDE_N}"
freq = RESAMPLING_FREQ

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--clean", type=bool, default=False,
                    help="Delete the crop folder if it exists. defaults to False")
args = parser.parse_args()



class JTK_Preprocess_Daily:
    def __init__(self, input_path, output_path, rs_freq) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.resampling_frequency = rs_freq
        self.wells = None
    
    def extract_well_names(self):
        well_paths = sorted(glob(f"{self.input_path}/*"))
        self.wells = [os.path.basename(well) for well in well_paths]
    
    def despike_and_crop_data(self, well_api):
        """
        Preprocess the labeled data to remove local spikes and crop it into daily windows

        Args:
            well_api (str): well API or ID
        
        Returns:
            None
        """
        esp_tests = sorted(glob(f"{self.input_path}/{well_api}/*.parquet"))

        for esp_test in esp_tests:
            df = pd.read_parquet(esp_test)
            esp_num = os.path.basename(esp_test).replace(".parquet","").split("_")[2]

            for data_date in np.unique(df.index.date):
                # Extract rows corresponding to the active day and save it
                daily_df = df[df.index.date == data_date]
                resample_daily_df = resample_and_interpolate_features(data_date,
                                                                      self.resampling_frequency,
                                                                      daily_df)
                
                # Check if the ESP pump was active (AC) or failed (PF)
                label_arr = resample_daily_df["Label"].to_numpy()
                label_class = "AC"
                if st.mode(label_arr).mode == 1:
                    label_class = "PF"
                
                save_path = f"{self.output_path}/{well_api}_esp#{esp_num}_{data_date}_{label_class}.npz"
                self.save_npz_file(resample_daily_df,save_path)


    def save_npz_file(self, interpolated_df, save_path):
        """
        Converts the dataframes to compressed numpy arrays in npz format

        Args:
            interpolated_df (pd.DataFrame): interpolated data that has features and labels
            save_path (str): file path to where the file should be saved
        
        Returns:
            None
        """

        features = interpolated_df.loc[:, interpolated_df.columns!="Label"].to_numpy()
        label = interpolated_df["Label"].to_numpy()

        np.savez_compressed(save_path, features=features, label=label, time=interpolated_df.index)
    
    def multiprocess_cropping(self):
        """
        Execute the despike_and_crop_data() function using multiple processors

        Returns:
            None
        """
        n_cpu = np.floor(cpu_count()/2).astype(int)

        with Pool(n_cpu) as p:
            with tqdm(total=len(self.wells), position=0) as pbar:
                for itr in p.imap_unordered(self.despike_and_crop_data, self.wells):
                    pbar.update()



if __name__ == "__main__":
    if os.path.exists(output_path) and args.clean:
        os.system(f"rm -rf {output_path}")
    
    if not os.path.exists(output_path):
        os.makedirs(f"{output_path}")
    
    crop = JTK_Preprocess_Daily(input_path, output_path, freq)
    crop.extract_well_names()
    # crop.despike_and_crop_data(crop.wells[0]) # single well test
    crop.multiprocess_cropping()
    os.system("rm -rf scripts/__pycache__")
