import numpy as np
from sklearn.preprocessing import StandardScaler
from app.DBmanager import getDataFrame

def addFeatures(df):
    g=9.80665
    df["elevacion_m"]= df["z"]/g
    df["wind_speed"]=np.sqrt(df["u10"]**2+df["v10"]**2)
    df["wind_dir"]=(np.degrees(np.arctan2(df["u10"],df["v10"]))+360)%360
    return df
def normaliceData(df,features,targets):
    scaler_x=StandardScaler()
    scaler_y=StandardScaler()
    X_Scaled=scaler_x.fit_transform(df[features])
    Y_Scaled=scaler_y.fit_transform(df[targets])
    return X_Scaled, Y_Scaled
def createWindows(X,Y,input_window=48,horizon=6):
    X_seq,Y_seq=[],[]
    for i in range(len(X)-input_window-horizon):
        X_seq.append(X[i: i+input_window])
        Y_seq.append(Y[i+input_window: i+input_window+horizon])
    return np.array(X_seq), np.array(Y_seq)