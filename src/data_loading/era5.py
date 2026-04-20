"""Descarga y transformación de datos ERA5.

Este módulo agrupa la lógica utilizada en el proyecto para descargar series
temporales horarias y campos estáticos de ERA5, procesar los archivos
resultantes y devolver estructuras listas para su almacenamiento en MongoDB.
"""

import os
import zipfile

import pandas as pd
from src.config import cds
import xarray as xr


class ERA5Loader:
    """Encapsula las operaciones de descarga y parsing de ERA5."""

    def __init__(self, cds_client=cds):
        """Inicializa el cargador con el cliente CDS configurado."""

        self.cds_client = cds_client

    def get_data_era5(self, start_dt, end_dt, lat, lon):
        """Descarga una serie temporal ERA5 para un punto de la rejilla."""

        dataset = "reanalysis-era5-single-levels-timeseries"
        date_str = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"
        request = {
            "variable": [
                "2m_dewpoint_temperature",
                "mean_sea_level_pressure",
                "surface_pressure",
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "location": {
                "latitude": lat,
                "longitude": lon,
            },
            "date": date_str,
            "data_format": "csv",
        }
        target_file = fr"C:\Users\User\Downloads\era5_wind_timeseries_{lat}_{lon}.zip"
        try:
            self.cds_client.retrieve(dataset, request, target_file)
            print(f"Datos descargados y guardados en {target_file}")
            json_data = self.convert_into_json(target_file)
            os.remove(target_file)
            return json_data
        except Exception as e:
            print(f"Error en el proceso de optenciÃ³n de los datos de ERA5:{e}")
            if os.path.exists(target_file):
                try:
                    os.remove(target_file)
                    print(f"El archivo {target_file} se ha borrado tras fallo")
                except Exception:
                    pass
            return {}

    def convert_into_json(self, target_file):
        """Convierte el ZIP descargado por ERA5 a lista de diccionarios."""

        try:
            with zipfile.ZipFile(target_file, "r") as z:
                csv_names = [name for name in z.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    print("No se encontrÃ³ ningÃºn CSV dentro del ZIP:", z.namelist())
                    return []

                csv_name = csv_names[0]
                print("Usando CSV dentro del ZIP:", csv_name)

                with z.open(csv_name) as f:
                    df = pd.read_csv(
                        f,
                        encoding="latin1",
                        comment="#",
                        engine="python",
                        sep=",",
                        on_bad_lines="skip",
                    )

            df = df.dropna(subset=["time"]) if "time" in df.columns else df
            data_dic = df.to_dict(orient="records")
            return data_dic
        except Exception as e:
            print(f"Error abriendo o procesando el ZIP/CSV: {e}")
            return []

    def get_static_fields(self):
        """Descarga los campos estáticos de ERA5 para la Península Ibérica."""

        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": ["geopotential", "land_sea_mask"],
            "year": "2024",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "area": [44, -10, 36, 4],
        }
        target_file = "iberia_static.grib"
        try:
            self.cds_client.retrieve(dataset, request, target_file)
            print(f"Campos estÃ¡ticos descargados en {target_file}")
            static_dict = self.build_static_dict(target_file)
            os.remove(target_file)
            return static_dict
        except Exception as e:
            print(f"Error en getStaticFields: {e}")
            if os.path.exists(target_file):
                try:
                    os.remove(target_file)
                except Exception:
                    pass
            return {}

    def build_static_dict(self, target_file):
        """Construye el diccionario de campos estáticos a partir de un GRIB."""

        ds = xr.open_dataset(target_file, engine="cfgrib")

        print(f"Variables disponibles: {list(ds.data_vars)}")

        if "z" not in ds.variables:
            raise ValueError("Variable 'z' no encontrada")
        if "lsm" not in ds.variables and "land_sea_mask" not in ds.variables:
            raise ValueError("Variable 'lsm'/'land_sea_mask' no encontrada")

        z = ds["z"]
        lsm = ds["lsm"] if "lsm" in ds.variables else ds["land_sea_mask"]

        static_dict = {}
        for lat in z.latitude.values:
            for lon in z.longitude.values:
                z_val = float(z.sel(latitude=lat, longitude=lon).values)
                lsm_val = float(lsm.sel(latitude=lat, longitude=lon).values)
                static_dict[(float(lat), float(lon))] = {
                    "z": z_val,
                    "lsm": lsm_val,
                }

        ds.close()
        print(f"Procesados {len(static_dict)} puntos (z + lsm)")
        return static_dict


def getDataERA5(start_dt, end_dt, lat, lon):
    """Mantiene la API histórica basada en función para la descarga puntual."""

    return ERA5Loader().get_data_era5(start_dt, end_dt, lat, lon)


def convertIntoJson(target_file):
    """Mantiene la API histórica basada en función para el parsing del ZIP."""

    return ERA5Loader().convert_into_json(target_file)


def getStaticFields():
    """Mantiene la API histórica basada en función para la descarga estática."""

    return ERA5Loader().get_static_fields()


def buildStaticDict(target_file):
    """Mantiene la API histórica basada en función para construir el diccionario."""

    return ERA5Loader().build_static_dict(target_file)
