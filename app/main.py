#imports

from app.era5 import getDataERA5,getGeoptencial
from app.aemet import getDataAemet
from datetime import datetime, timedelta
from app.DBmanager import loadIntoDB,getDataFrame,saveZDictMongo
from app.DFmanager import addFeatures
import torch
from app.models.tft_model import buildTFTDataSet, buildTFTModel,trainTFT
import numpy as np



#funciones
def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates
def loadData(start,end,lats,lons):
    current_start=start
    control=1
    while current_start<=end:
        #gestión del bucle
        current_end= current_start + timedelta(days=15)
        if current_end>end:
            current_end=end
        #obtención y carga de los datos
        dates= setDates(current_start,current_end)
        for lat in lats:
            for lon in lons:
                data=getDataERA5(dates["era5"][0],dates["era5"][1],lat,lon)
                loadIntoDB(data,control)
        #siguiente fecha
        current_start=current_end+timedelta(days=1)
#variables
start=datetime(2024,1,1)
end=datetime(2024,1,3)
lats = np.arange(36.0, 44.0 + 0.001, 0.25)
lons = np.arange(-10.0, 4.0 + 0.001, 0.25)
#z_dict = getGeoptencial()
#código
#saveZDictMongo(z_dict)
loadData(start,end,lats,lons)
#df=getDataFrame()
#df=addFeatures(df)

#targets = ["wind_speed", "wind_dir_sin","wind_dir_cos"]
#static_reals = ["latitude", "longitude", "elevacion_m"]
#time_varying_known_reals = ["time_idx"]
#time_varying_unknown_reals = ["u10", "v10", "t2m", "d2m", "msl", "sp","tcwv", "cape", "blh","wind_speed", "wind_dir"]
#max_encoder_length=48
#max_prediction_length=6
#batch_size=64
#max_epochs=30

#training=buildTFTDataSet(df,targets,static_reals,time_varying_known_reals,time_varying_unknown_reals,max_encoder_length,max_prediction_length)
#tft=buildTFTModel(training)
#trainTFT(training,tft)