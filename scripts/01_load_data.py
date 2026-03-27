"""Script de carga de ERA5 y campos estáticos en MongoDB.

Este punto de entrada ejecuta la fase de ingesta del proyecto: comprueba si los
campos estáticos ya están almacenados y, a continuación, lanza la descarga de
los datos horarios ERA5 para el rango temporal configurado.
"""

import logging

from src.config import PARAMS
from src.data_loading.era5 import getStaticFields
from src.db.DBmanager import DBmanager
from src.utils import loadData


class DataLoadScript:
    """Orquesta la fase de carga inicial de datos ERA5."""

    def __init__(self):
        """Inicializa el logger y el acceso a MongoDB."""

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.db_manager = DBmanager()

    def run(self):
        """Ejecuta la secuencia completa de carga de datos."""

        static_dict = self.db_manager.loadStaticDictMongo()
        if not static_dict:
            self.logger.info("Calculando campos estÃ¡ticos...")
            static_dict = getStaticFields()
            self.db_manager.saveStaticDictMongo(static_dict)

        self.logger.info("Cargando datos ERA5...")
        loadData(
            self.db_manager,
            PARAMS["data"]["start_dt"],
            PARAMS["data"]["end_dt"],
            PARAMS["data"]["lats"],
            PARAMS["data"]["lons"],
            max_workers=3,
        )


if __name__ == "__main__":
    DataLoadScript().run()
