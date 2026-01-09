#imports

from app.era5 import getDataERA5,getGeoptencial
from app.aemet import getDataAemet
from datetime import datetime, timedelta
from app.DBmanager import loadIntoDB,getDataFrame,saveZDictMongo
from app.DFmanager import addFeatures
import torch
from app.models.tft_model import buildTFTDataSet, buildTFTModel,trainTFT
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time



#funciones
def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates
def process_point_period(start_dt, end_dt, lat, lon, control=1):
    dates = setDates(start_dt, end_dt)
    data = getDataERA5(dates["era5"][0], dates["era5"][1], lat, lon)
    if not data:
        print(f"Sin datos para punto {lat}, {lon} entre {start_dt} y {end_dt}")
        return
    loadIntoDB(data, control)
    print(f"Punto {lat}, {lon} cargado para {start_dt}–{end_dt}")
def loadData(start,end,lats,lons,max_workers=3):
    current_start=start
    control=1
    while current_start<=end:
        #gestión del bucle
        current_end= current_start + timedelta(days=15)
        if current_end>end:
            current_end=end
        print(f"Procesando bloque {current_start}–{current_end}")
        futures=[]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for lat in lats:
                for lon in lons:
                    futures.append(
                        executor.submit(
                            process_point_period,
                            current_start,
                            current_end,
                            float(lat),
                            float(lon),
                            control
                        )
                    )
                    time.sleep(0.3) #para espaciar envíos a CDS
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print ("Error en un punto:",e)
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