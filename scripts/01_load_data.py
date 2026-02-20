"""Carga datos ERA5 y orografía."""
import logging
from src.config import PARAMS
from src.data_loading.era5 import getStaticFields
from src.db.DBmanager import DBmanager
from src.utils import loadData

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    db_manager = DBmanager()
    
    static_dict = db_manager.loadStaticDictMongo()
    if not static_dict:
        logger.info("Calculando campos estáticos...")
        static_dict = getStaticFields()
        db_manager.saveStaticDictMongo(static_dict)
    
    logger.info("Cargando datos ERA5...")
    loadData(
        db_manager,
        PARAMS["data"]["start_dt"],
        PARAMS["data"]["end_dt"],
        PARAMS["data"]["lats"],
        PARAMS["data"]["lons"],
        max_workers=3
    )