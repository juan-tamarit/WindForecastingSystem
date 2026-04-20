"""Utilidades de orquestación para la ingesta de ERA5.

Este módulo conserva la lógica usada por el proyecto para recorrer el dominio
espacial y temporal, descargar cada bloque de ERA5 y almacenarlo en MongoDB.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from threading import Semaphore

from src.data_loading.era5 import getDataERA5


class ERA5IngestionService:
    """Orquesta la carga de ERA5 por bloques temporales y puntos de rejilla."""

    def set_dates(self, fecha_ini_dt, fecha_fin_dt):
        """Devuelve el rango temporal en los formatos usados por el proyecto."""

        fecha_ini_aemet = fecha_ini_dt.strftime("%Y-%m-%d") + "T00:00:00UTC"
        fecha_fin_aemet = fecha_fin_dt.strftime("%Y-%m-%d") + "T23:59:59UTC"
        dates = {
            "aemet": [fecha_ini_aemet, fecha_fin_aemet],
            "era5": [fecha_ini_dt, fecha_fin_dt],
        }
        return dates

    def process_point_period(self, start_dt, end_dt, lat, lon, db_manager, cds_semaphore, control):
        """Descarga y guarda un bloque temporal para un punto de la rejilla."""

        lat_r, lon_r = round(lat, 2), round(lon, 2)

        last_date = db_manager.get_last_date_for_point(lat, lon)
        if last_date and last_date >= start_dt:
            print(f"Punto {lat_r}, {lon_r} ya procesado hasta {last_date}")
            return
        try:
            with cds_semaphore:
                data = getDataERA5(start_dt, end_dt, lat, lon)
            if data:
                db_manager.loadIntoDB(data, control)
                print(f"{lat_r}, {lon_r} cargado")
        except Exception as e:
            print(f"Error {lat_r}, {lon_r}: {e}")

    def load_data(self, db_manager, start, end, lats, lons, max_workers=3):
        """Recorre el rango temporal en bloques mensuales y lanza las descargas."""

        cds_semaphore = Semaphore(3)
        current_start = start
        control = 1
        while current_start <= end:
            current_end = min(current_start + timedelta(days=31), end)
            print(f"Procesando bloque {current_start}â€“{current_end}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.process_point_period,
                        current_start,
                        current_end,
                        float(lat),
                        float(lon),
                        db_manager,
                        cds_semaphore,
                        control,
                    )
                    for lat in lats
                    for lon in lons
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print("Error en un punto:", e)

            current_start = current_end + timedelta(days=1)


def setDates(fecha_ini_dt, fecha_fin_dt):
    """Mantiene la API histórica basada en función para el rango temporal."""

    return ERA5IngestionService().set_dates(fecha_ini_dt, fecha_fin_dt)


def processPointPeriod(start_dt, end_dt, lat, lon, db_manager, cds_semaphore, control):
    """Mantiene la API histórica basada en función para un punto y periodo."""

    return ERA5IngestionService().process_point_period(
        start_dt,
        end_dt,
        lat,
        lon,
        db_manager,
        cds_semaphore,
        control,
    )


def loadData(db_manager, start, end, lats, lons, max_workers=3):
    """Mantiene la API histórica basada en función para la carga por bloques."""

    return ERA5IngestionService().load_data(
        db_manager,
        start,
        end,
        lats,
        lons,
        max_workers=max_workers,
    )
