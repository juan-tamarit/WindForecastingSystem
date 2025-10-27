import xarray as xr
import json
import os
from config import cds

def getDataERA5():
    #Esto es solo una pequeña prueba para comprobar que se conecta a la api, hay que modificarlo para que pueda ser algo general
    dataset = 'reanalysis-era5-single-levels'
    request = {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind'
        ],
        'year': '2024',
        'month': '01',
        'day': '01',
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf'
    }
    target_file = r"C:\Users\User\Downloads\era5_wind_10m.nc"
    cds.retrieve(dataset, request, target_file)
    print(f"Datos descargados y guardados en {target_file}")
    json_data=convertIntoJson(target_file)
    #borramos el archivo una vez procesado y convertido en json
    os.remove(target_file)
    return json_data
def convertIntoJson(target_file):
    #leer el archivo netcdf con xarray para convertirlo en json
    ds=xr.open_dataset(target_file,decode_times=False)
    data_dic=ds.to_dict() #Conviertir en diccionario
    json_data = json.dumps(data_dic, indent=2)  # Serializar a JSON
    #devolvemos la información
    return json_data