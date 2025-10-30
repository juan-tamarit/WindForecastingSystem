#imports

from app.era5 import getDataERA5 
from app.aemet import getDataAemet
import datetime



#funciones
def setFecha():
    #Se creo originalmente para sacar datos de la antártida, podría tener utilidad para sacar datos históricos de AEMET
    fechaStr=input("El formato de la fecha debe ser el siguiente: AAAA-MM-DDTHH:MM:SS")
    try:
        fecha=datetime.datetime.strptime(fechaStr,"%Y-%m-%dT%H:%M:%S")
        return fecha.strftime("%Y-%m-%dT%H:%M:%SUTC")
    except ValueError:
        raise ValueError("La fecha no tiene el formato AAAA-MM-DDTHH:MM:SS o no es válida")

#variables
start_year=2020
end_year=2026
#código
#dataAEMET=getDataAemet("2025-01-01T00:00:00UTC","2025-01-16T23:59:59UTC")
dataAEMET=getDataAemet()
print(dataAEMET)
#dataERA5=getDataERA5(start_year,end_year)
