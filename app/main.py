#imports
import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import warnings #el asunto de los warnings
from sklearn.exceptions import DataConversionWarning #el asunto de los warnings
from app.era5 import getDataERA5,getGeoptencial
from app.aemet import getDataAemet
from datetime import datetime, timedelta
from app.DBmanager import loadIntoDB,getDataFrame,saveZDictMongo
from app.DFmanager import addFeatures,splitDataFrame
from app.metricas import evaluateTarget,plotErrorHistogram,plotPredictions
from app.models.tft_model import buildTFTDataSet, buildValidation,buildTFTModel,trainTFT,loadBestModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE



#funciones

# -----------------------------------------------------------------------------
# Construcción de rangos temporales coherentes para AEMET y ERA5
#
# Objetivo:
# - A partir de dos datetimes (inicio y fin) generar:
#     · Un rango de fechas en formato string para las peticiones a AEMET
#     · Un rango de datetimes "crudos" para las peticiones a ERA5
#
# Salida:
# - Diccionario con dos entradas:
#   1) "aemet": [fecha_ini_AEMET, fecha_fin_AEMET]
#      - Strings en formato: "YYYY-MM-DDT00:00:00UTC" / "YYYY-MM-DDT23:59:59UTC"
#      - Cubren el día completo en UTC para las APIs de AEMET.
#   2) "era5": [fecha_ini_dt, fecha_fin_dt]
#      - Datetimes originales, usados directamente en las peticiones ERA5
#        (que ya trabajan con resolución horaria). [web:1][web:183]
#
# Notas:
# - Mantener una única función de fechas ayuda a sincronizar exactamente los
#   periodos que se piden a AEMET y ERA5, evitando desalineaciones de días.
# -----------------------------------------------------------------------------

def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates

# -----------------------------------------------------------------------------
# Procesamiento de un punto y periodo (unidad de trabajo paralelizable)
#
# Esta función define la "tarea mínima" que se ejecutará en paralelo:
#  - Calcula las fechas efectivas a solicitar para ERA5 (setDates)
#  - Llama a getDataERA5 para el punto (lat, lon) y el rango [start_dt, end_dt]
#  - Si se reciben datos, los inserta en Mongo mediante loadIntoDB
#
# Parámetros:
# - start_dt, end_dt: rango temporal a procesar para un punto
# - lat, lon: coordenadas del punto de rejilla ERA5
# - control: selector de rama en loadIntoDB (1 para ERA5)
#
# Uso en paralelo:
# - Esta función es la que se pasa a ThreadPoolExecutor.submit()
#   en la versión paralela del bucle sobre lats × lons
# -----------------------------------------------------------------------------

def processPointPeriod(start_dt, end_dt, lat, lon, control=1):
    dates = setDates(start_dt, end_dt)
    data = getDataERA5(dates["era5"][0], dates["era5"][1], lat, lon)
    if not data:
        print(f"Sin datos para punto {lat}, {lon} entre {start_dt} y {end_dt}")
        return
    loadIntoDB(data, control)
    print(f"Punto {lat}, {lon} cargado para {start_dt}–{end_dt}")

# -----------------------------------------------------------------------------
# Versión paralela del cargador ERA5 (ThreadPoolExecutor)
#
# Objetivo:
# - Acelerar la carga de medio año / un año de ERA5 sobre una rejilla de puntos,
#   manteniendo un número razonable de peticiones simultáneas al CDS.
#
# Estrategia:
# - Divide el periodo global [start, end] en bloques de ~15 días (como loadData)
# - Para cada bloque, lanza tareas process_point_period en paralelo con un
#   ThreadPoolExecutor limitado por max_workers
# - Espacia ligeramente el envío de tareas para no saturar la cola del CDS
#
# Parámetros:
# - start, end: rango temporal global de la carga
# - lats, lons: rejilla de latitudes/longitudes
# - max_workers: máximo de peticiones concurrentes al CDS (recomendado 3)
#
# Notas:
# - Usa as_completed para capturar errores por punto sin detener el resto
# - Diseñado para convivir con el diccionario z (orografía) ya persistido en Mongo
# -----------------------------------------------------------------------------

def loadData(start,end,lats,lons,max_workers=3):
    current_start=start
    control=1
    while current_start<=end:
        #gestión del bucle
        current_end= current_start + timedelta(days=15)
        if current_end>end:
            current_end=end
        print(f"Procesando bloque {current_start}–{current_end}")
        #hilos
        futures=[]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for lat in lats:
                for lon in lons:
                    futures.append(
                        executor.submit(
                            processPointPeriod,
                            current_start,
                            current_end,
                            float(lat),
                            float(lon),
                            control
                        )
                    )
                    time.sleep(0.3) #para espaciar envíos a CDS
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print ("Error en un punto:",e)
        #siguiente ventana temporal
        current_start=current_end+timedelta(days=1)


def main():
    """
    #Carga de datos en mongoDB

    start=datetime(2024,2,29)
    end=datetime(2024,3,31)
    lats = np.arange(36, 44.0 + 0.001, 0.25)
    lons = np.arange(-10, 4.0 + 0.001, 0.25)
    z_dict = getGeoptencial()

    código

    saveZDictMongo(z_dict)
    loadData(start,end,lats,lons)
    """
    df=getDataFrame()
    df=addFeatures(df)

    targets = ["wind_speed", "wind_dir_sin","wind_dir_cos"]
    static_reals = ["latitude", "longitude", "elevacion_m"]
    time_varying_known_reals = ["time_idx"]
    time_varying_unknown_reals = ["wind_speed", "wind_dir_sin", "wind_dir_cos","u10","v10","u100","v100","t2m","d2m","skt","sp","msl","tp","ssrd","strd"]
    max_encoder_length=24 #para probar
    max_prediction_length=1 #para probar
    batch_size=128#para probar
    max_epochs=1#para probar

    print(df.columns)
    print(df.dtypes[["valid_time", "time_idx", "location_id"]])
    print(df[["wind_speed", "wind_dir_sin", "wind_dir_cos"]].describe())

    train_fact=0.8
    df_train,df_val=splitDataFrame(df,train_fact)
    training=buildTFTDataSet(df_train,targets,static_reals,time_varying_known_reals,time_varying_unknown_reals,max_encoder_length,max_prediction_length)
    validation=buildValidation(training,df_val)
    tft=buildTFTModel(training)
    checkpoint_callback=trainTFT(training,validation,tft,batch_size,max_epochs)
    best_checkpoint_path=checkpoint_callback.best_model_path
    best_tft=loadBestModel(best_checkpoint_path)

    #Calcular métricas de evaluación sobre df_val:
    for i, name in enumerate(targets):
        metrics = evaluateTarget(best_tft, validation, target_idx=i, batch_size=batch_size)
        print(f"== {name} ==")
        print(f"  MAE  : {metrics['MAE']:.4f}")
        print(f"  RMSE : {metrics['RMSE']:.4f}")
        print(f"  MAPE : {metrics['MAPE']:.2f}%")
    #Crear visualizaciones:

    plotPredictions(best_tft, validation, batch_size, 3)

    # Histogramas de error por target
    for i, name in enumerate(targets):
        print(f"Mostrando histograma de errores para {name}")
        plotErrorHistogram(best_tft, validation, batch_size, i)

warnings.filterwarnings("ignore",message="X does not have valid feature names, but StandardScaler was fitted with feature names") #el asunto de los warnings
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # opcional pero recomendado en Windows
    main()