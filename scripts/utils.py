import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

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


def resample_and_interpolate_features(doi,frq,daily_df):
    """
    Resample training features to desired frequency

    Args:
        doi (datetime): date of interest
        frq (int): frequency values for interpolation. Unit is minutes
        daily_df (pd.DataFrame): raw dataframe within the time window of each ESP

    Returns:
        pd.DataFrame: interpolated dataframe
    """

    # Specify interpolation rows
    start_time = f"{str(doi)} 00:00:00"
    end_time = f"{str(doi)} 23:59:59"
    date_range = pd.date_range(start=start_time, end=end_time, freq=f"{frq}min")

    # Create dummy array dimensions
    nrows,ncols = len(date_range), len(daily_df.columns)
    tmp_array = np.zeros((nrows,ncols)) * np.nan

    # Loop through columns and linearly interpolate non-nan rows with data
    for col_idx,col in enumerate(daily_df.columns):
        tmp_df = daily_df[[col]].dropna()

        if len(tmp_df) > 2:
            # Fit data
            float_times = tmp_df.index.to_numpy().astype(float)
            linear_fit = interp1d(float_times,tmp_df[col])

            # Interpolate data
            data_dt_limit = date_range[(date_range>=tmp_df.index[0]) & 
                                    (date_range<=tmp_df.index[-1])]
            data_dt_idx = np.where((date_range>=tmp_df.index[0]) & 
                                (date_range<=tmp_df.index[-1]))[0]
            float_data_dt_limit = data_dt_limit.to_numpy().astype(float)
            intrp_data = linear_fit(float_data_dt_limit)

            # Populate dummy array
            tmp_array[data_dt_idx, col_idx] = intrp_data
    
    # Create interpolated dataframe
    intrp_daily_df = pd.DataFrame(tmp_array)
    intrp_daily_df.columns = daily_df.columns
    intrp_daily_df.index = date_range
    # nan values are filled in the labels because label values are known
    intrp_daily_df["Label"] = intrp_daily_df["Label"].bfill().ffill().to_numpy()

    return intrp_daily_df



def replace_nan(data, replacement_mode='zero', axis=None):
    """
    Replace NaN values in a NumPy array with a specific value, handling all-NaN slices.

    Args:
        data (np.ndarray): The input NumPy array.
        replacement_mode (str): The mode for replacing NaN values. Choices are 'zero', 'mean', or 'median'.
        axis (int or tuple, optional): The axis or axes along which to calculate the mean or median. If None, the mean or median is computed over the entire array.

    Returns:
        np.ndarray: The NumPy array with NaN values replaced, ensuring no all-NaN slice warnings.
    """
    if replacement_mode == 'zero':
        return np.nan_to_num(data, nan=0.0)
    elif replacement_mode in ['mean', 'median']:
        data = np.copy(data)  # Work on a copy of the data to avoid modifying the original array
        
        if replacement_mode == 'mean':
            replacement_value = np.nanmean(data, axis=axis, keepdims=True)
        else:  # 'median'
            replacement_value = np.nanmedian(data, axis=axis, keepdims=True)
        
        # Handle the all-NaN slices by replacing them with a specific value (e.g., 0)
        # This is a simple workaround to avoid the warning and fill those slices
        if np.isnan(replacement_value).any():
            replacement_value[np.isnan(replacement_value)] = 0.0  # You can choose a different fallback value if necessary
        
        np.putmask(data, np.isnan(data), replacement_value)
        return data
    else:
        raise ValueError("Invalid replacement mode. Choose 'zero', 'mean', or 'median'.")



def normalize_timeseries(array):
    """
    Normalize an array of tensors

    Args:
        array (torch.Tensor): 2D array of Tensors. Rows is timestamps, columns are features

    Returns:
        array (torch.Tensor): Normalized array along each feature column
    """
    # Feature transformations
    mean = array.mean(dim=0)
    std = array.std(dim=0)

    # Avoid division by zero
    std[std == 0] = 1

    # Normalize the array
    norm_arr = (array - mean) / std
    return norm_arr