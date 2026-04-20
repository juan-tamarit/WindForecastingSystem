"""Construccion del conjunto de datos procesado.

Este modulo define la clase encargada de recuperar observaciones ERA5 desde
MongoDB, convertirlas a ``DataFrame`` y generar las variables derivadas
utilizadas durante el entrenamiento y la evaluacion.
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient

from src.config import MDB


class DFmanager:
    """Gestiona la transicion entre datos crudos y datos procesados."""

    def __init__(self):
        """Inicializa el acceso a las colecciones relevantes del proyecto."""

        self.client = MongoClient(MDB["uri"])
        self.db = self.client[MDB["db_name"]]
        self.collection_era = self.db[MDB["collection_era"]]
        self.collection_pro = self.db[MDB["collection_pro"]]

    def getCollectionPro(self):
        """Devuelve la coleccion Mongo que almacena los datos procesados."""

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
        """Anade las variables derivadas necesarias para el modelado."""

        g = np.float32(9.80665)
        df["elevacion_m"] = df["z"].astype(np.float32) / g
        base_time = df["valid_time"].min()
        df["time_idx"] = ((df["valid_time"] - base_time).dt.total_seconds() // 3600).astype(np.int32)
        df["location_id"] = df.groupby(["latitude", "longitude"], sort=False, observed=True).ngroup().astype(np.int32)
        return df

    def get_normalization_stats(self):
        """Calcula medias y desviaciones tipicas globales sobre ``collection_pro``."""

        print("--- Calculando estadisticas de normalizacion en collection_pro ---")

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
                raise ValueError("La coleccion 'collection_pro' esta vacia o no existe.")

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
                    print(f"Aviso: Desviacion estandar de {var} es 0. Ajustada a 1.0.")

            print("Estadisticas obtenidas con exito.")
            return stats

        except Exception as e:
            print(f"Error al calcular estadisticas en MongoDB: {e}")
            return None

    def get_spatial_stats(self, times, lats, lons):
        """Calcula climatologia espacial por punto de rejilla para los tiempos dados."""

        print("--- Calculando climatologia espacial en collection_pro ---")

        if not times:
            return None

        lat_map = {round(float(lat), 2): i for i, lat in enumerate(lats)}
        lon_map = {round(float(lon), 2): j for j, lon in enumerate(lons)}

        mean_fields = {
            "2t": np.zeros((len(lats), len(lons)), dtype=np.float32),
            "10u": np.zeros((len(lats), len(lons)), dtype=np.float32),
            "10v": np.zeros((len(lats), len(lons)), dtype=np.float32),
            "msl": np.zeros((len(lats), len(lons)), dtype=np.float32),
        }
        std_fields = {
            "2t": np.ones((len(lats), len(lons)), dtype=np.float32),
            "10u": np.ones((len(lats), len(lons)), dtype=np.float32),
            "10v": np.ones((len(lats), len(lons)), dtype=np.float32),
            "msl": np.ones((len(lats), len(lons)), dtype=np.float32),
        }

        pipeline = [
            {"$match": {"valid_time": {"$in": times}}},
            {
                "$group": {
                    "_id": {"latitude": "$latitude", "longitude": "$longitude"},
                    "avg_2t": {"$avg": "$t2m"},
                    "std_2t": {"$stdDevPop": "$t2m"},
                    "avg_10u": {"$avg": "$u10"},
                    "std_10u": {"$stdDevPop": "$u10"},
                    "avg_10v": {"$avg": "$v10"},
                    "std_10v": {"$stdDevPop": "$v10"},
                    "avg_msl": {"$avg": "$msl"},
                    "std_msl": {"$stdDevPop": "$msl"},
                }
            },
        ]

        try:
            results = list(self.collection_pro.aggregate(pipeline))

            if not results:
                raise ValueError("No se han encontrado resultados para la climatologia espacial.")

            for row in results:
                lat = round(float(row["_id"]["latitude"]), 2)
                lon = round(float(row["_id"]["longitude"]), 2)
                i = lat_map.get(lat)
                j = lon_map.get(lon)

                if i is None or j is None:
                    continue

                mean_fields["2t"][i, j] = np.float32(row.get("avg_2t", 0.0) or 0.0)
                mean_fields["10u"][i, j] = np.float32(row.get("avg_10u", 0.0) or 0.0)
                mean_fields["10v"][i, j] = np.float32(row.get("avg_10v", 0.0) or 0.0)
                mean_fields["msl"][i, j] = np.float32(row.get("avg_msl", 0.0) or 0.0)

                std_fields["2t"][i, j] = np.float32(row.get("std_2t", 1.0) or 1.0)
                std_fields["10u"][i, j] = np.float32(row.get("std_10u", 1.0) or 1.0)
                std_fields["10v"][i, j] = np.float32(row.get("std_10v", 1.0) or 1.0)
                std_fields["msl"][i, j] = np.float32(row.get("std_msl", 1.0) or 1.0)

            stats = {
                var: {
                    "mean": mean_fields[var],
                    "std": std_fields[var],
                }
                for var in mean_fields
            }

            print("Climatologia espacial obtenida con exito.")
            return stats

        except Exception as e:
            print(f"Error al calcular climatologia espacial en MongoDB: {e}")
            return None
