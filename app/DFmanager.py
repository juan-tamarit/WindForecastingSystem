import numpy as np

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
    df["time_idx"] = ((df["valid_time"] - df["valid_time"].min()).dt.total_seconds() // 3600).astype(int)
    df["location_id"] = df.groupby(["latitude", "longitude"]).ngroup()
    return df

# -----------------------------------------------------------------------------
# División temporal del DataFrame en entrenamiento y validación
#
# Objetivo:
# - Separar el histórico en dos subconjuntos respetando el orden temporal,
#   de forma que la validación use siempre datos posteriores al entrenamiento.
#
# Criterio de división:
# - Se toma el máximo índice temporal disponible (max_time_idx).
# - training_cutoff = int(max_time_idx * train_fact)
#   · train_fact = 0.8 → 80 % del rango temporal para entrenamiento,
#     20 % final para validación.
# - Todas las filas con time_idx <= training_cutoff se usan para entrenar.
# - Todas las filas con time_idx  > training_cutoff se reservan para validar.
#
# Ventajas:
# - Evita fuga de información (no se entrena con datos que luego se validan).
# - Es coherente con problemas de forecasting donde el tiempo tiene una dirección
#   clara y no se debe barajar aleatoriamente el conjunto de datos. [web:355]
# -----------------------------------------------------------------------------

def splitDataFrame(df,train_fact):
    max_time_idx = df["time_idx"].max()
    training_cutoff = int(max_time_idx * train_fact)
    df_train = df[df["time_idx"] <= training_cutoff]
    df_val = df[df["time_idx"] > training_cutoff]
    return df_train,df_val