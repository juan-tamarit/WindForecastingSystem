#imports

from app.era5 import getDataERA5 
from app.aemet import getDataAemet
from datetime import datetime, timedelta
from app.DBmanager import loadIntoDB,getDataFrame
import numpy as np
from sklearn.preprocessing import StandardScaler

#funciones
def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates
def loadData(start,end):
    current_start=start
    control=1
    while current_start<=end:
        #gestión del bucle
        current_end= current_start + timedelta(days=15)
        if current_end>end:
            current_end=end
        #obtención y carga de los datos AEMET
        dates= setDates(current_start,current_end)
        data=getDataERA5(dates["era5"][0],dates["era5"][1])
        loadIntoDB(data,control)
        #siguiente fecha
        current_start=current_end+timedelta(days=1)
def createWindows(X,Y,input_window=48,horizon=6):
    X_seq,Y_seq=[],[]
    for i in range(len(X)-input_window-horizon):
        X_seq.append(X[i: i+input_window])
        Y_seq.append(Y[i+input_window: i+input_window+horizon])
    return np.array(X_seq), np.array(Y_seq)
#variables
start=datetime(2024,1,1)
end=datetime(2024,1,16)
g=9.80665
#código
#loadData(start,end)
df=getDataFrame()

#Creación de variables a partir de datos de Mongo
df["elevacion_m"]= df["z"]/g
df["wind_speed"]=np.sqrt(df["u10"]**2+df["v10"]**2)
df["wind_dir"]=(np.degrees(np.arctan2(df["u10"],df["v10"]))+360)%360
#Columnas que se utilizaran como input y que deseamos como output
features = ["u10","v10","t2m","d2m","msl","sp","tcwv","cape","blh","latitude","longitude","elevacion_m","wind_speed","wind_dir"]
targets=["wind_speed","wind_dir"]
#Normalización de los datos
scaler_x=StandardScaler()
scaler_y=StandardScaler()
X_Scaled=scaler_x.fit_transform(df[features])
Y_Scaled=scaler_y.fit_transform(df[targets])
#Ventanas de tiempo
X_seq,Y_seq=createWindows(X_Scaled,Y_Scaled)