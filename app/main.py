#imports

from app.era5 import getDataERA5 
from app.aemet import getDataAemet
from datetime import datetime, timedelta
from app.DBmanager import loadIntoDB,getDataFrame
from app.DFmanager import addFeatures
import torch


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
#variables
start=datetime(2024,1,1)
end=datetime(2024,1,16)
#código
#loadData(start,end)
df=getDataFrame()
df=addFeatures(df)

targets = ["wind_speed", "wind_dir"]
static_reals = ["latitude", "longitude", "elevacion_m"]
time_varying_known_reals = ["time_idx"]
time_varying_unknown_reals = ["u10", "v10", "t2m", "d2m", "msl", "sp","tcwv", "cape", "blh","wind_speed", "wind_dir"]