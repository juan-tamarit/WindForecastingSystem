import xarray as xr
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
            # Viento en superficie
            '10m_u_component_of_wind',                 # componente viento Este-Oeste
            '10m_v_component_of_wind',                 # componente viento Norte-Sur
            '10m_wind_gust_since_previous_post_processing',  # rachas de viento a 10 m
            # Termodinámica en superficie
            '2m_temperature',                          # temperatura a 2 m
            '2m_dewpoint_temperature',                 # punto de rocío a 2 m
            'surface_pressure',                        # presión a nivel de superficie
            'mean_sea_level_pressure',                 # presión a nivel medio del mar
            'relative_humidity',                       # humedad relativa
            # Nubes / radiación / precipitación
            'cloud_cover',                             # cobertura nubosa total
            'total_precipitation',                     # precipitación total acumulada
            'surface_solar_radiation_downwards',       # radiación solar en superficie
            # Estructura vertical / estabilidad
            'boundary_layer_height',                   # altura de la capa límite
            'convective_available_potential_energy',   # CAPE
            'convective_inhibition',                   # CIN
            'total_column_water_vapour'               # agua precipitable total
        ],
        'year': years,
        'month': months,
        'day': days,
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        'area': [44.0, -10.0, 36.0, 4.0],
        'format': 'grib'
    }


    target_file = r"C:\Users\User\Downloads\era5_wind_10m.grib"
    try:
        cds.retrieve(dataset, request, target_file)
        print(f"Datos descargados y guardados en {target_file}")
        json_data=convertIntoJson(target_file)
        #os.remove(target_file)#borramos el archivo una vez procesado y convertido en json
        return json_data
    except Exception as e:
        print (f"Error en el proceso de optención de los datos de ERA5:{e}")
        if os.path.exists(target_file):
            try:
                #os.remove(target_file)
                print(f"El archivo {target_file} se ha borrado tras fallo")
            except Exception:
                pass
        return None
def convertIntoJson(target_file):
    #leer el archivo netcdf con xarray para convertirlo en json
    try:
        ds=xr.open_dataset(target_file,engine="cfgrib",decode_times=False)
        df=ds.to_dataframe().reset_index()
        data_dic=df.to_dict(orient="records") #Conviertir en diccionario
        return data_dic
    except Exception as e:
        print(f"Error abriendo o procesando el archivo NetCDF: {e}")
        raise