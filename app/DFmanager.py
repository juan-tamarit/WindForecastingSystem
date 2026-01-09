import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Enriquecimiento del DataFrame con variables derivadas para el modelo
#
# Objetivo:
# - Añadir features físicas y estructurales a partir de las variables ERA5,
#   preparando el DataFrame para el modelado de series temporales (p.ej. TFT).
#
# Features calculadas:
# 1) elevacion_m:
#    - Altura geométrica aproximada del terreno en metros.
#    - Se obtiene dividiendo el geopotencial z entre la gravedad estándar g.
#      elevacion_m = z / 9.80665  [web:137]
#
# 2) wind_speed:
#    - Módulo del viento a 10 m a partir de las componentes u10 y v10:
#      wind_speed = sqrt(u10² + v10²)  [web:125]
#
# 3) wind_dir:
#    - Dirección del viento en grados (0–360), calculada a partir de (u10, v10):
#      wind_dir = (atan2(u10, v10) en grados + 360) % 360
#    - Se usa atan2(u10, v10) para respetar el cuadrante correcto.
#
# 4) wind_dir_sin / wind_dir_cos:
#    - Proyección de la dirección del viento en el círculo unitario:
#      · wind_dir_sin = sin(wind_dir en radianes)
#      · wind_dir_cos = cos(wind_dir en radianes)
#    - Evitan la discontinuidad 0/360 al alimentar modelos numéricos.
#
# 5) time_idx:
#    - Índice temporal entero relativo, en horas, desde el primer timestamp:
#      time_idx = (valid_time - min(valid_time)) / 3600
#    - Útil como time_idx para librerías de forecasting (p.ej. PyTorch Forecasting).
#
# 6) location_id:
#    - Identificador entero de cada serie espacial (lat, lon):
#      · Se agrupa por (latitude, longitude)
#      · ngroup() asigna un id consecutivo a cada par único
#    - Permite manejar múltiples series (una por gridpoint) en un solo modelo.
#
# Retorno:
# - El mismo DataFrame de entrada, enriquecido con las columnas nuevas.
# -----------------------------------------------------------------------------

def addFeatures(df):
    g=9.80665
    df["elevacion_m"]= df["z"]/g
    df["wind_speed"]=np.sqrt(df["u10"]**2+df["v10"]**2)
    df["wind_dir"]=(np.degrees(np.arctan2(df["u10"],df["v10"]))+360)%360
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir"]))
    df["time_idx"] = ((df["valid_time"] - df["valid_time"].min()) // 3600).astype(int)
    df["location_id"] = df.groupby(["latitude", "longitude"]).ngroup()
    return df