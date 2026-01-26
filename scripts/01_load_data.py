"""Carga datos ERA5 y orografía."""
import logging
from src.config import PARAMS
from src.data_loading.era5 import getGeoptencial  
from src.db.DBmanager import saveZDictMongo,loadZDictMongo
from src.utils import loadData # Directo, sin cambios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Verificando carga de datos ERA5")
    
    # Orografía: SOLO si colección vacía
    z_dict = loadZDictMongo()
    if not z_dict:  # Diccionario vacío → calcular y guardar
        logger.info("Orografía no encontrada, calculando getGeoptencial()...")
        z_dict = getGeoptencial()
        saveZDictMongo(z_dict)
        logger.info("Orografía guardada en MongoDB")
    else:
        logger.info("Orografía ya existe en MongoDB, saltando getGeoptencial()")
    
    # ERA5: siempre carga (o añade check similar si quieres)
    logger.info("Cargando datos ERA5...")
    loadData(
        PARAMS["data"]["start_dt"],
        PARAMS["data"]["end_dt"],
        PARAMS["data"]["lats"],
        PARAMS["data"]["lons"]
    )
    logger.info("Carga completada")