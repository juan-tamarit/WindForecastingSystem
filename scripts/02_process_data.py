"""Script de construcción de la colección de datos procesados.

Este punto de entrada toma los datos ERA5 ya cargados en MongoDB, identifica
las observaciones nuevas respecto a ``datosProcesados`` y genera las variables
derivadas empleadas por el entrenamiento.
"""

import logging

from src.frame.DFmanager import DFmanager


class DataProcessingScript:
    """Orquesta la generación incremental de la colección procesada."""

    def __init__(self):
        """Inicializa el logger y el gestor de dataframes."""

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.df_manager = DFmanager()

    def run(self):
        """Ejecuta la construcción incremental de ``datosProcesados``."""

        self.logger.info("Construyendo datosProcesados...")

        if self.df_manager.getCollectionPro().count_documents({}) > 0:
            pipeline = [{"$sort": {"valid_time": -1}}, {"$limit": 1}]
            last_doc = list(self.df_manager.getCollectionPro().aggregate(pipeline))[0]
            after_date = last_doc["valid_time"]
        else:
            after_date = None
            self.logger.info("Primera carga")

        df_raw_new = self.df_manager.getDataFrame(after_date)
        self.logger.info(f"Datos raw nuevos: {df_raw_new.shape}")

        if len(df_raw_new) == 0:
            print("No hay datos nuevos")
        else:
            df_proc_new = self.df_manager.addFeatures(df_raw_new)
            docs = df_proc_new.to_dict(orient="records")
            self.df_manager.getCollectionPro().insert_many(docs)
            print(f"Insertados {len(docs)} documentos nuevos")


if __name__ == "__main__":
    DataProcessingScript().run()
