import httpx
import json
from config import api_key

def getDataAemet(fecha_ini=None,fecha_fin=None):
    try:
        if fecha_ini is None and fecha_fin is None:
            url = f"https://opendata.aemet.es/opendata/api/observacion/convencional/todas"
        else:
            url = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_ini}/fechafin/{fecha_fin}/todasestaciones"
        end_point = httpx.get(f"{url}?api_key={api_key}", timeout=300.0)
        end_point.raise_for_status()
        url_datos = end_point.json()["datos"]
        datos = httpx.get(f"{url_datos}?api_key={api_key}", timeout=300.0)
        datos.raise_for_status()
        json_data=transformDataIntoJson(datos)
        return json_data
    except httpx.RequestError as exc:
        print(f"Error en la conexión al intentar acceder a {exc.request.url}->{exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error HTTP {exc.status_code} en la url {exc.request.url}")
    except ValueError:
        print("La respuesta no tiene un formato JSON válido")
def transformDataIntoJson(data):
    try:
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
        else:
            print(f"Tipo de contenido inesperado: {content_type}")
    except UnicodeDecodeError as ude:
        print(f"Error en la decodificación de caracteres: {ude}")
        return None
    except json.JSONDecodeError as jde:
        print(f"Error en la decodificación del JSON: {jde}")
        print(f"Texto recibido:",decoded_text[:500])
        return None
    except Exception as e:
        print(f"Error inesperado en transformDataIntoJson: {e}")
        return None