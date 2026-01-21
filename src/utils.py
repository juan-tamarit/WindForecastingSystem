#imports
import time
from src.data.era5 import getDataERA5
from datetime import timedelta
from src.db.DBmanager import loadIntoDB
from concurrent.futures import ThreadPoolExecutor, as_completed
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
