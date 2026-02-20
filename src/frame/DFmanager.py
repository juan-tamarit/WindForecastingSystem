import numpy as np
from pymongo import MongoClient
import pandas as pd
from src.config import MDB
import torch
#index in collection_era db.datosEra5.createIndex({"valid_time": 1})
#index in collection_pro db.datosProcesados.createIndex({"location_id": 1, "time_idx": 1})
#index in collection_pro db.datosProcesados.createIndex({"datetime": 1})
#index in collection_pro db.datosProcesados.createIndex({"time_idx": 1}, {background: true})
class DFmanager:
    def __init__(self):
        self.client=MongoClient(MDB["uri"])
        self.db=self.client[MDB["db_name"]]
        self.collection_era=self.db[MDB["collection_era"]]
        self.collection_pro=self.db[MDB["collection_pro"]]
    def getCollectionPro(self):return self.collection_pro
    def getDataFrame(self,after_date=None)->pd.DataFrame:       
        pipeline = [{"$match": {"valid_time": {"$gt": after_date} if after_date else {}}},
                {"$sort": {"valid_time": 1}},
                {"$project": {"_id": 0, "valid_time": 1, "z": 1, "latitude": 1, "longitude": 1, 
                              "t2m": 1, "u10": 1, "v10": 1, "msl": 1, "sp": 1, "d2m": 1, "lsm": 1}}]

        data = list(self.collection_era.aggregate(pipeline))
    
        if not data:
            return pd.DataFrame()
    
        df = pd.DataFrame(data)
        df["valid_time"] = pd.to_datetime(df["valid_time"], cache=True, errors='coerce')  # Cache para fechas repetidas
        return df.reset_index(drop=True)

    def addFeatures(self,df:pd.DataFrame)-> pd.DataFrame:
        g = np.float32(9.80665)
        df["elevacion_m"] = (df["z"].astype(np.float32) / g)
        base_time = df["valid_time"].min()
        df["time_idx"] = ((df["valid_time"] - base_time).dt.total_seconds() // 3600).astype(np.int32)
        df["location_id"] = df.groupby(["latitude", "longitude"], sort=False, observed=True).ngroup().astype(np.int32)
        return df