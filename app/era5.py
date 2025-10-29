import xarray as xr
import json
import os
from config import cds

def getDataERA5(start_year,end_year):
    dataset = 'reanalysis-era5-single-levels'
    #Generar lista de fechas con todos los días entre 01/01/star_year y 31/12/end_year-1 
    years = [str(y) for y in range(start_year, end_year)]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    #creamos el request
    request = {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind'
        ],
        'year': years,
        'month': months,
        'day': days,
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf'
    }
    target_file = r"C:\Users\User\Downloads\era5_wind_10m.nc"
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
        return None
def convertIntoJson(target_file):
    #leer el archivo netcdf con xarray para convertirlo en json
    try:
        ds=xr.open_dataset(target_file,decode_times=False)
        data_dic=ds.to_dict() #Conviertir en diccionario
        json_data = json.dumps(data_dic, indent=2)  # Serializar a JSON
        return json_data#devolvemos la información
    except Exception as e:
        print(f"Error abriendo o procesando el archivo NetCDF: {e}")
        raise