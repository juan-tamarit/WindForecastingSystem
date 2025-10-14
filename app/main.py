#imports

import httpx
import datetime
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
def getData(x):
    try:
        url=setUrl(x)
        endPoint= httpx.get(f"{url}?api_key={api_key}",timeout=300.0)
        endPoint.raise_for_status()
        url_datos=endPoint.json()["datos"]
        datos= httpx.get(f"{url_datos}?api_key={api_key}",timeout=300.0)
        datos.raise_for_status()
        return datos
    except httpx.RequestError as exc:
        print (f"Error en la conexión al intentar acceder a {exc.request.url}->{exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error HTTP {exc.status_code} en la url {exc.request.url}")
    except ValueError:
        print(f"La respuesta no tiene un formato JSON válido")
#variables
x=int(input("1: tiempo actual\n2: tiempo entre dos fechas pasadas"))
data=getData(x)
print (data)