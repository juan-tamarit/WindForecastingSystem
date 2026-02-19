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

    def splitDataFrame(self, df: pd.DataFrame, train_fact: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = int(df["time_idx"].max() * train_fact)
        mask_train = df["time_idx"] <= cutoff
        df_train = df[mask_train].copy()
        df_val = df[~mask_train].copy()
        return df_train,df_val

    def getProcessedDataFrame(self, date, mode) -> pd.DataFrame:
        if mode == 0:
            pipeline = [
                {"$match": {"valid_time": {"$lte": date} if date else {}}},
                {"$sort": {"location_id": 1, "time_idx": 1}},
                {"$project": {"_id": 0,
                    "time_idx": 1,
                    "valid_time": 1,
                    "location_id": 1,
                    "latitude": 1,
                    "longitude": 1,
                    "elevacion_m": 1,
                    "lsm": 1,
                    "t2m": 1,
                    "u10": 1,
                    "v10": 1,
                    "msl": 1,
                    "sp": 1,
                    "d2m": 1,
                    "z": 1}}
            ]
        elif mode == 1:
            pipeline = [
            {"$match": {"valid_time": {"$gte": date} if date else {}}},
            {"$sort": {"location_id": 1, "time_idx": 1}},
            {"$project": {"_id": 0,
                    "time_idx": 1,
                    "valid_time": 1,
                    "location_id": 1,
                    "latitude": 1,
                    "longitude": 1,
                    "elevacion_m": 1,
                    "lsm": 1,
                    "t2m": 1,
                    "u10": 1,
                    "v10": 1,
                    "msl": 1,
                    "sp": 1,
                    "d2m": 1,
                    "z": 1}}
            ]
        data = list(self.collection_pro.aggregate(pipeline))
        return pd.DataFrame(data).reset_index(drop=True)

    def load_processed_era5_aurora(self,df: pd.DataFrame) -> tuple[dict,dict]:    
        batch_data = {
            "2t": df["t2m"].astype(np.float32).values,
            "10u": df["u10"].astype(np.float32).values,
            "10v": df["v10"].astype(np.float32).values,
            "msl": (df["msl"] / 100).astype(np.float32).values,
            "sp": (df["sp"] / 100).astype(np.float32).values,
            "2d": df["d2m"].astype(np.float32).values,
            "lsm": df["lsm"].astype(np.float32).values,
            "z": df["z"].astype(np.float32).values
        }
        static_df = df.groupby("location_id")[["latitude", "longitude", "elevacion_m"]].first()
        static_vars = {
            "location_ids": df["location_id"].unique(),
            "latitudes": static_df["latitude"].values,
            "longitudes": static_df["longitude"].values,
            "elevations": static_df["elevacion_m"].values
        }
        return batch_data, static_vars
    
    def get_statics_only(self):
        # 1. Obtenemos listas únicas y ordenadas de coordenadas
        # Esto define los ejes de tu "imagen" de la Península
        lats = sorted(self.collection_pro.distinct("latitude"), reverse=True) # Norte -> Sur
        lons = sorted(self.collection_pro.distinct("longitude"))             # Oeste -> Este
    
        n_lats = len(lats)
        n_lons = len(lons)
    
        # 2. Creamos la matriz de elevación vacía
        elevation_map = torch.zeros((n_lats, n_lons), dtype=torch.float32)
    
        # Mapas de búsqueda rápida para índices
        lat_to_idx = {lat: i for i, lat in enumerate(lats)}
        lon_to_idx = {lon: j for j, lon in enumerate(lons)}
    
        # 3. Poblamos la matriz
        # Traemos solo un documento por cada punto (usando location_id)
        cursor = self.collection_pro.find({}, {"latitude": 1, "longitude": 1, "elevacion_m": 1, "_id": 0})
    
        # Usamos un set para no procesar el mismo punto miles de veces (una por cada hora)
        processed_points = set()
    
        for doc in cursor:
            lat, lon = doc["latitude"], doc["longitude"]
            point = (lat, lon)
        
            if point not in processed_points:
                i = lat_to_idx[lat]
                j = lon_to_idx[lon]
                elevation_map[i, j] = doc["elevacion_m"]
                processed_points.add(point)
            
            # Si ya tenemos todos los puntos del mapa, paramos para ahorrar tiempo
            if len(processed_points) == (n_lats * n_lons):
                break

        # 4. Aurora también suele necesitar las coordenadas en formato grid (lat/lon para cada píxel)
        lat_grid, lon_grid = torch.meshgrid(torch.tensor(lats), torch.tensor(lons), indexing='ij')

        return {
            "elevation": elevation_map, # [Lat, Lon]
            "lat_grid": lat_grid,       # [Lat, Lon]
            "lon_grid": lon_grid,       # [Lat, Lon]
            "n_lats": n_lats,
            "n_lons": n_lons
        }