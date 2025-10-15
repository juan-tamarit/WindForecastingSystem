#imports

import httpx
import datetime
import json
from config import api_key

#funciones
def setFecha():
    fechaStr=input("El formato de la fecha debe ser el siguiente: AAAA-MM-DDTHH:MM:SS")
    try:
        fecha=datetime.datetime.strptime(fechaStr,"%Y-%m-%dT%H:%M:%S")
        return fecha.strftime("%Y-%m-%dT%H:%M:%SUTC")
    except ValueError:
        raise ValueError("La fecha no tiene el formato AAAA-MM-DDTHH:MM:SS o no es válida")
def setUrl(x):
    if(x==1):
        return f"https://opendata.aemet.es/opendata/api/observacion/convencional/todas"
    elif(x==2):
        fechaIniStr=setFecha()
        fechaFinStr=setFecha()
        return f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fechaIniStr}/fechafin/{fechaFinStr}/todasestaciones"
import json

def getData(x):
    try:
        url = setUrl(x)
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
#variables
x=int(input("1: tiempo actual\n2: tiempo entre dos fechas pasadas"))
data=getData(x)
transformed_data=transform_data_into_json(data)