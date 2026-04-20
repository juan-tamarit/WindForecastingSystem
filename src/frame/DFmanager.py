"""Construcción del conjunto de datos procesado.

Este módulo define la clase encargada de recuperar observaciones ERA5 desde
MongoDB, convertirlas a ``DataFrame`` y generar las variables derivadas
utilizadas durante el entrenamiento y la evaluación.
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient

from src.config import MDB


class DFmanager:
    """Gestiona la transición entre datos crudos y datos procesados."""

    def __init__(self):
        """Inicializa el acceso a las colecciones relevantes del proyecto."""

        self.client = MongoClient(MDB["uri"])
        self.db = self.client[MDB["db_name"]]
        self.collection_era = self.db[MDB["collection_era"]]
        self.collection_pro = self.db[MDB["collection_pro"]]

    def getCollectionPro(self):
        """Devuelve la colección Mongo que almacena los datos procesados."""

        return self.collection_pro

    def getDataFrame(self, after_date=None):
        """Recupera observaciones ERA5 y las devuelve como ``DataFrame``."""

        pipeline = [
            {"$match": {"valid_time": {"$gt": after_date} if after_date else {}}},
            {"$sort": {"valid_time": 1}},
            {
                "$project": {
                    "_id": 0,
                    "valid_time": 1,
                    "z": 1,
                    "latitude": 1,
                    "longitude": 1,
                    "t2m": 1,
                    "u10": 1,
                    "v10": 1,
                    "msl": 1,
                    "sp": 1,
                    "d2m": 1,
                    "lsm": 1,
                }
            },
        ]

        data = list(self.collection_era.aggregate(pipeline))

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["valid_time"] = pd.to_datetime(df["valid_time"], cache=True, errors="coerce")
        return df.reset_index(drop=True)

    def addFeatures(self, df):
        """Añade las variables derivadas necesarias para el modelado."""

        g = np.float32(9.80665)
        df["elevacion_m"] = df["z"].astype(np.float32) / g
        base_time = df["valid_time"].min()
        df["time_idx"] = ((df["valid_time"] - base_time).dt.total_seconds() // 3600).astype(np.int32)
        df["location_id"] = df.groupby(["latitude", "longitude"], sort=False, observed=True).ngroup().astype(np.int32)
        return df

    def get_normalization_stats(self):
        """Calcula medias y desviaciones típicas sobre ``collection_pro``."""

        print("--- Calculando estadÃ­sticas de normalizaciÃ³n en collection_pro ---")

        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_2t": {"$avg": "$t2m"},
                    "std_2t": {"$stdDevPop": "$t2m"},
                    "avg_10u": {"$avg": "$u10"},
                    "std_10u": {"$stdDevPop": "$u10"},
                    "avg_10v": {"$avg": "$v10"},
                    "std_10v": {"$stdDevPop": "$v10"},
                    "avg_msl": {"$avg": "$msl"},
                    "std_msl": {"$stdDevPop": "$msl"},
                }
            }
        ]

        try:
            results = list(self.collection_pro.aggregate(pipeline))

            if not results:
                raise ValueError("La colecciÃ³n 'collection_pro' estÃ¡ vacÃ­a o no existe.")

            res = results[0]
            stats = {
                "2t": {"mean": float(res["avg_2t"]), "std": float(res["std_2t"])},
                "10u": {"mean": float(res["avg_10u"]), "std": float(res["std_10u"])},
                "10v": {"mean": float(res["avg_10v"]), "std": float(res["std_10v"])},
                "msl": {"mean": float(res["avg_msl"]), "std": float(res["std_msl"])},
            }

            for var, val in stats.items():
                if val["std"] == 0 or val["std"] is None:
                    stats[var]["std"] = 1.0
                    print(f"Aviso: DesviaciÃ³n estÃ¡ndar de {var} es 0. Ajustada a 1.0.")

            print("EstadÃ­sticas obtenidas con Ã©xito.")
            return stats

        except Exception as e:
            print(f"Error al calcular estadÃ­sticas en MongoDB: {e}")
            return None
