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
    def get_normalization_stats(self):
        """
        Calcula media y desviación estándar directamente en MongoDB (collection_pro).
        Se usa para la normalización Z-score en el entrenamiento de Aurora.
        """
        print("--- Calculando estadísticas de normalización en collection_pro ---")
        
        # Pipeline optimizado para las variables que Aurora necesita
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_2t":  {"$avg": "$t2m"},
                "std_2t":  {"$stdDevPop": "$t2m"},
                "avg_10u": {"$avg": "$u10"},
                "std_10u": {"$stdDevPop": "$u10"},
                "avg_10v": {"$avg": "$v10"},
                "std_10v": {"$stdDevPop": "$v10"},
                "avg_msl": {"$avg": "$msl"},
                "std_msl": {"$stdDevPop": "$msl"}
            }}
        ]
        
        try:
            # Ejecutamos sobre la colección procesada (la que usa el DataModule)
            results = list(self.collection_pro.aggregate(pipeline))
            
            if not results:
                raise ValueError("La colección 'collection_pro' está vacía o no existe.")
                
            res = results[0]
            stats = {
                "2t":  {"mean": float(res["avg_2t"]),  "std": float(res["std_2t"])},
                "10u": {"mean": float(res["avg_10u"]), "std": float(res["std_10u"])},
                "10v": {"mean": float(res["avg_10v"]), "std": float(res["std_10v"])},
                "msl": {"mean": float(res["avg_msl"]), "std": float(res["std_msl"])}
            }
            
            # Limpieza de seguridad: evitar divisiones por cero si los datos son constantes
            for var, val in stats.items():
                if val["std"] == 0 or val["std"] is None:
                    stats[var]["std"] = 1.0
                    print(f"Aviso: Desviación estándar de {var} es 0. Ajustada a 1.0.")
            
            print("Estadísticas obtenidas con éxito.")
            return stats

        except Exception as e:
            print(f"Error al calcular estadísticas en MongoDB: {e}")
            return None