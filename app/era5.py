import pandas as pd
import json
import zipfile
import os
from datetime import timedelta,datetime
from config import cds
import xarray as xr

# -----------------------------------------------------------------------------
# Descarga de series temporales ERA5 para un punto (dataset time-series)
#
# Esta función encapsula la lógica de:
#  - Construir la petición al dataset "reanalysis-era5-single-levels-timeseries"
#  - Solicitar datos horarios para un rango [start_dt, end_dt]
#  - Incluir múltiples variables físicas relevantes para viento
#  - Descargar el ZIP (CSV interno) en disco y convertirlo a lista de dicts
#
# Parámetros:
# - start_dt, end_dt: datetimes que definen el rango temporal (incluyendo ambas fechas)
# - lat, lon: coordenadas del punto ERA5 (coinciden con la rejilla del time-series)
#
# Notas:
# - data_format = 'csv' produce un ZIP con un único CSV interno
# - convertIntoJson se encarga de abrir el ZIP, leer el CSV y devolver la lista de
#   documentos listos para insertar en MongoDB
# - En caso de error se devuelve una lista vacía para evitar romper el flujo
# -----------------------------------------------------------------------------

def getDataERA5(start_dt,end_dt,lat,lon):
    dataset = 'reanalysis-era5-single-levels-timeseries'
    date_str = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"
    #creamos el request
    request = {
    "variable": [
        "2m_dewpoint_temperature",
        "mean_sea_level_pressure",
        "skin_temperature",
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "2m_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "geopotential"
    ],
    'location': {
        'latitude': lat,
        'longitude': lon
    },
    'date':date_str,
    'data_format': 'csv'
}
    target_file = fr"C:\Users\User\Downloads\era5_wind_timeseries_{lat}_{lon}.zip"
    try:
        cds.retrieve(dataset, request, target_file)
        print(f"Datos descargados y guardados en {target_file}")
        json_data=convertIntoJson(target_file)
        os.remove(target_file)#borramos el archivo una vez procesado y convertido en json
        return json_data
    except Exception as e:
        print (f"Error en el proceso de optención de los datos de ERA5:{e}")
        if os.path.exists(target_file):
            try:
                os.remove(target_file)
                print(f"El archivo {target_file} se ha borrado tras fallo")
            except Exception:
                pass
        return {}

# -----------------------------------------------------------------------------
# Conversión del ZIP/CSV ERA5 a lista de documentos (para MongoDB)
#
# Esta función:
#  - Abre el archivo ZIP generado por la API de ERA5 time-series
#  - Localiza el primer fichero CSV contenido en el ZIP
#  - Lee el CSV usando pandas con una configuración tolerante:
#      * encoding='latin1' para evitar problemas de decodificación
#      * comment='#' para ignorar líneas de metadatos/comentarios
#      * engine='python' + on_bad_lines='skip' para saltar filas problemáticas
#  - Elimina filas sin campo temporal 'time' (si existe esa columna)
#  - Devuelve una lista de diccionarios (una entrada por fila del CSV)
#
# Uso:
# - El resultado puede pasarse directamente a loadIntoDB para su inserción en MongoDB
# - No se realizan transformaciones físicas aquí (solo parsing de archivo)
# -----------------------------------------------------------------------------

def convertIntoJson(target_file):
    try:
        # 1) Abrir el ZIP
        with zipfile.ZipFile(target_file, 'r') as z:
            # buscar el primer CSV dentro del ZIP
            csv_names = [name for name in z.namelist() if name.lower().endswith('.csv')]
            if not csv_names:
                print("No se encontró ningún CSV dentro del ZIP:", z.namelist())
                return []

            csv_name = csv_names[0]
            print("Usando CSV dentro del ZIP:", csv_name)

            # 2) Leer el CSV directamente desde el ZIP
            with z.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    encoding='latin1',   # o la que haga falta, o sin encoding si no falla
                    comment='#',
                    engine='python',
                    sep=',',
                    on_bad_lines='skip'
                )

        # 3) Convertir a lista de dicts
        df = df.dropna(subset=['time']) if 'time' in df.columns else df
        data_dic = df.to_dict(orient='records')
        return data_dic
    except Exception as e:
        print(f"Error abriendo o procesando el ZIP/CSV: {e}")
        return []
    
def getGeoptencial():
    dataset = 'reanalysis-era5-single-levels'
    request={
        'product_type': 'reanalysis',
            'variable': ['geopotential'],
            'year': '2024',
            'month': '01',
            'day': '01',
            'time': '00:00',
            'area': [44, -10, 36, 4],
            'format': 'netcdf',
    }
    target_file='iberia_geopotencial.nc'
    try:
        cds.retrieve(dataset, request, target_file)
        print(f"Datos Geopotenciales descargados y guardados en {target_file}")
        z_dic=buildZDict(target_file)
        os.remove(target_file)#borramos el archivo una vez procesado y convertido en json
        return z_dic
    except Exception as e:
        print (f"Error en el proceso de optención de los datos geopotenciales de ERA5:{e}")
        if os.path.exists(target_file):
            try:
                os.remove(target_file)
                print(f"El archivo {target_file} se ha borrado tras fallo")
            except Exception:
                pass
        return {}
    
def buildZDict(target_file):
    ds = xr.open_dataset(target_file)
    z = ds['z']
    z_dict = {}
    for lat in z.latitude.values:
        for lon in z.longitude.values:
            z_val = float(z.sel(latitude=lat, longitude=lon))
            z_dict[(float(lat), float(lon))] = z_val
    return z_dict