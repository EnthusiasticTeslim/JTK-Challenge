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
        str_date = pd.to_datetime(f"{m}/{d}/{y[-2:]} {t}", format="%m/%d/%y %H:%M")
    return str_date