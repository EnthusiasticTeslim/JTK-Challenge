import pandas as pd
import numpy as np

def date_parser(str_date):
    """
    Cleans up the date time columns in the esp_tracker file to
    render them compatible with python.

    Args:
        str_date (str): date of interest

    Returns:
        pd.Timestamp: pandas compatible datetime
    """
    if str_date is not None and str_date is not np.nan:
        str_date = str_date.replace(";",":")
        try:
            dy,t = str_date.split(" ")
        except:
            dy,t = str_date.split("  ")
        m,d,y = dy.split("/")
        str_date = pd.to_datetime(f"{m}/{d}/{y[-2:]} {t}", 
                                  format="%m/%d/%y %H:%M")
    return str_date



def clip_zscore_outliers(df, threshold=3):
    """
    Replaces spikes/outliers in the dataset with the mean.
    Outliers are identified using zscore method.

    Args:
        df (pd.DataFrame): Training DataFrame
        threshold (int): Number of sd from from the mean. Defaults to 3.

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    feat_cols = df.columns.tolist()
    feat_cols.remove("Label")
    df_feat = df[feat_cols]
    z_scores = df_feat.apply(lambda x: (x - x.mean()) / x.std())
    clipped_df = df_feat.mask(abs(z_scores) > threshold, df_feat.mean(), axis=0)
    clipped_df["Label"] = df["Label"]
    return clipped_df



def resample_dataframe(df, duration):
    """
    Resample training dataset to a specified frequency using
    linear interpolation.

    Args:
        df (pd.DataFrame): Training DataFrame
        duration (int): Resampling frequency in minutes

    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    df = df.resample(f"{duration}min").interpolate(method="linear")
    df = df.map(lambda x: round(x, 4))
    return df