import numpy as np
from sklearn.preprocessing import StandardScaler

def addFeatures(df):
    g=9.80665
    df["elevacion_m"]= df["z"]/g
    df["wind_speed"]=np.sqrt(df["u10"]**2+df["v10"]**2)
    df["wind_dir"]=(np.degrees(np.arctan2(df["u10"],df["v10"]))+360)%360
    df["time_idx"] = ((df["valid_time"] - df["valid_time"].min()) // 3600).astype(int)
    df["location_id"] = df.groupby(["latitude", "longitude"]).ngroup()
    return df