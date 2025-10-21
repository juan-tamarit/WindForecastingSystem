#imports

import httpx
import datetime
import json
from config import api_key, cds

#funciones
def setFecha():
    #Se creo originalmente para sacar datos de la antártida, podría tener utilidad para sacar datos históricos de AEMET
    fechaStr=input("El formato de la fecha debe ser el siguiente: AAAA-MM-DDTHH:MM:SS")
    try:
        fecha=datetime.datetime.strptime(fechaStr,"%Y-%m-%dT%H:%M:%S")
        return fecha.strftime("%Y-%m-%dT%H:%M:%SUTC")
    except ValueError:
        raise ValueError("La fecha no tiene el formato AAAA-MM-DDTHH:MM:SS o no es válida")
def getDataAemet():
    try:
        url = f"https://opendata.aemet.es/opendata/api/observacion/convencional/todas"
        endPoint = httpx.get(f"{url}?api_key={api_key}", timeout=300.0)
        endPoint.raise_for_status()
        url_datos = endPoint.json()["datos"]
        datos = httpx.get(f"{url_datos}?api_key={api_key}", timeout=300.0)
        datos.raise_for_status()
        return datos
    except httpx.RequestError as exc:
        print(f"Error en la conexión al intentar acceder a {exc.request.url}->{exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error HTTP {exc.status_code} en la url {exc.request.url}")
    except ValueError:
        print("La respuesta no tiene un formato JSON válido")
def transform_data_into_json(data):
    #con esta función nos aseguramos de que los datos recibidos desde la api de aemet sean json
    content_type=data.headers.get('content-type','').lower()
    charset='iso-8859-15'
    #en caso de que el codificado no fuera iso-8859-15 lo sacamos del header
    if 'charset=' in content_type:
        charset=content_type.split('charset=')[-1]
    decoded_text=data.content.decode(charset)
    #de acuerdo con la documentación la respuesta debería ser directamente aplication/json pero todas las veces que hemos intentado sacarlo era 'text/plain'
    if 'aplication/json' in content_type:
        return data.json()
    elif 'text/plain' in content_type:
        json_objects=[json.loads(decoded_text)]
        return json_objects
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

#variables
#data=getDataAemet()
#transformed_data=transform_data_into_json(data)
#print(transformed_data)

