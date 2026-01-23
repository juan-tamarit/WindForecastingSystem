from pymongo import MongoClient
from src.db.DBmanager import collection_pro
from src.frame.DFmanager import getDataFrame, addFeatures
import logging

if __name__ == "__main__":
    logging.info("Construyendo datosProcesados...")
    
    # 1. Fecha última procesada
    if collection_pro.count_documents({}) > 0:
        pipeline = [{"$sort": {"valid_time": -1}}, {"$limit": 1}]
        last_doc = list(collection_pro.aggregate(pipeline))[0]
        after_date = last_doc["valid_time"]
    else:
        after_date = None
        logging.info("Primera carga")
    
    # 2. Solo datos nuevos desde Mongo (filtro eficiente)
    df_raw_new = getDataFrame(after_date)
    logging.info(f"Datos raw nuevos: {df_raw_new.shape}")
    
    if len(df_raw_new) == 0:
        print("No hay datos nuevos")
    else:
        # 3. Procesar solo nuevos
        df_proc_new = addFeatures(df_raw_new)
        docs = df_proc_new.to_dict(orient="records")
        collection_pro.insert_many(docs)
        print(f"Insertados {len(docs)} documentos nuevos")