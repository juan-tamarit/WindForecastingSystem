#imports
import time
from src.data_loading.era5 import getDataERA5
from datetime import timedelta
from src.db.DBmanager import DBmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import List


def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates

def processPointPeriod(start_dt, end_dt, lat: float, lon: float, db_manager, cds_semaphore,control):
    lat_r, lon_r = round(lat, 2), round(lon, 2)
    
    # Check last date to avoid reprocessing
    last_date = db_manager.get_last_date_for_point(lat, lon)
    if last_date and last_date >= start_dt:
        print(f"Punto {lat_r}, {lon_r} ya procesado hasta {last_date}")
        return
    try:
        with cds_semaphore:
            data = getDataERA5(start_dt, end_dt, lat, lon)
        if data:
            db_manager.loadIntoDB(data,control)
            print(f"{lat_r}, {lon_r} cargado")
    except Exception as e:
        print(f"Error {lat_r}, {lon_r}: {e}")

def loadData(db_manager,start,end,lats,lons,max_workers=3):
    cds_semaphore=Semaphore(3)
    current_start=start
    control=1
    while current_start<=end:
        #gestión del bucle
        current_end= min(current_start + timedelta(days=31),end)
        print(f"Procesando bloque {current_start}–{current_end}")
        #hilos
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    processPointPeriod,
                    current_start, 
                    current_end,
                    float(lat), 
                    float(lon),
                    db_manager, 
                    cds_semaphore, 
                    control
                )
                for lat in lats 
                for lon in lons
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print ("Error en un punto:",e)
        #siguiente ventana temporal
        current_start=current_end+timedelta(days=1)
