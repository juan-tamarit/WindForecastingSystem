import xarray as xr
import json
import os
from datetime import timedelta,datetime
from config import cds

def getDataERA5(start_dt,end_dt):
    dataset = 'reanalysis-era5-single-levels'
    #calcular días entre fecha_ini y fecha_end
    date_list=[start_dt+timedelta(days=d) for d in range((end_dt-start_dt).days)]
    #Generar lista
    years = sorted(list(set([int(dt.strftime("%Y")) for dt in date_list])))
    months = sorted(list(set([int(dt.strftime("%m")) for dt in date_list])))
    days = sorted(list(set([int(dt.strftime("%d")) for dt in date_list])))
    #creamos el request
    request = {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind',# viento Este-Oeste a 10 m
            '10m_v_component_of_wind',# viento Norte-Sur a 10 m
            '2m_temperature',# temperatura a 2 m en kelvin
            'surface_pressure'# presión atmosférica
        ],
        'year': years,
        'month': months,
        'day': days,
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': [44.0, -10.0, 36.0, 4.0],
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
        df=ds.to_dataframe().reset_index()
        data_dic=df.to_dict(orient="records") #Conviertir en diccionario
        return data_dic
    except Exception as e:
        print(f"Error abriendo o procesando el archivo NetCDF: {e}")
        raise