#imports

from app.era5 import getDataERA5 
from app.aemet import getDataAemet
from datetime import datetime
from pymongo import MongoClient
import json
from config import mdb
#funciones
def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates
#variables
start=datetime(2024,1,1)
end=datetime(2024,1,16)
dates=setDates(start,end)
#código
client=MongoClient(mdb["uri"])
db=client[mdb["db_name"]]
coleccion=db[mdb["collection_name"]]

#dataAEMET=getDataAemet(dates["aemet"][0],dates["aemet"][1])
#dataAEMET=getDataAemet()
#dataERA5=getDataERA5(dates["era5"][0],dates["era5"][1])
#coleccion.insert_many(dataERA5)
