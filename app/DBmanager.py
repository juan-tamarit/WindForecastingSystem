from pymongo import MongoClient
from config import mdb
from datetime import datetime
import pandas as pd

client=MongoClient(mdb["uri"])
db=client[mdb["db_name"]]
coleccion=db[mdb["collection_name"]]
collection_aemet=db[mdb["collection_aemet"]]
collection_era=db[mdb["collection_era"]]
collection_z=db[mdb["collection_z"]]

# -----------------------------------------------------------------------------
# Carga de datos en MongoDB (ERA5, AEMET u otros)
#
# Esta función centraliza la inserción en distintas colecciones según "control":
#  - control == 0: datos AEMET (observaciones)
#      * Convierte 'fecha' (string) a datetime
#      * Inserta documentos en collection_aemet
#
#  - control == 1: datos ERA5 time-series
#      * Convierte 'valid_time' (string) a datetime
#      * Enriquecimiento con geopotencial z (si existe z_dict en Mongo):
#          · Usa latitude / longitude del doc
#          · Redondea coordenadas para casarlas con la rejilla del diccionario
#          · Añade campo 'z' al documento si hay coincidencia
#      * Inserta documentos en collection_era
#
#  - en cualquier otro caso: inserción masiva directa en "coleccion"
#
# Comportamiento en errores:
# - Envuelve toda la lógica en try/except
# - Muestra un mensaje de error y relanza la excepción para depuración
# -----------------------------------------------------------------------------

def loadIntoDB(data,control):
    try:
        z_dict=loadZDictMongo()
        if control==0:
            for doc in data:
                doc['fecha']=datetime.strptime(doc['fecha'],"%Y-%m-%d")
                collection_aemet.insert_one(doc)
        elif control==1:
            for doc in data:
                doc['valid_time']=datetime.strptime(doc['valid_time'],"%Y-%m-%d %H:%M:%S")
                if z_dict:
                    lat = float(doc['latitude'])
                    lon = float(doc['longitude'])
                    lat = round(lat, 2); lon = round(lon, 2)
                    z_val = z_dict.get((lat, lon))
                    if z_val is not None:
                        doc['z'] = z_val
                collection_era.insert_one(doc)
        else:
            coleccion.insert_many(data)
    except Exception as e:
        print(f"Error en la carga: {e}")
        raise
def saveZDictMongo(z_dict):
    docs = []
    for (lat, lon), z_val in z_dict.items():
        docs.append({
            "latitude": lat,
            "longitude": lon,
            "z": z_val
        })
    if docs:
        collection_z.insert_many(docs)
def loadZDictMongo():
    z_dict = {}
    for doc in collection_z.find({}, {"_id": 0, "latitude": 1, "longitude": 1, "z": 1}):
        lat = float(doc["latitude"])
        lon = float(doc["longitude"])
        z_dict[(lat, lon)] = float(doc["z"])
    return z_dict
def getDataFrame():
    data=list(collection_era.find({}))
    df=pd.DataFrame(data)
    #eliminar variables que no necesitamos
    df=df.drop(columns=["_id","time"])
    #aseguramos el formato del timestamp
    df["valid_time"]=pd.to_datetime(df["valid_time"])
    #ordenamos el dataframe
    df=df.sort_values(by="valid_time").reset_index(drop=True)
    return df