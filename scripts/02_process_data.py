from pymongo import MongoClient
from src.db.DBmanager import collection_pro
from src.frame.DFmanager import getDataFrame, addFeatures
import logging

if __name__ == "__main__":
    logging.info("Construyendo datosProcesados...")
    
    # 1. Datos raw existentes
    df_raw = getDataFrame()
    
    # 2. Procesar
    df_proc = addFeatures(df_raw)
    
    # 3. Incremental: solo nuevos datos
    if collection_pro.count_documents({}) > 0:
        last_time_proc = list(collection_pro.find().sort("valid_time", -1).limit(1))[0]["valid_time"]
        df_new = df_proc[df_proc["valid_time"] > last_time_proc]
        logging.info(f"Nuevos datos: {len(df_new)}")
    else:
        df_new = df_proc
        logging.info(f"Primera carga: {len(df_new)}")
    
    # 4. Insertar solo nuevos
    if len(df_new) > 0:
        docs = df_new.to_dict(orient="records")
        collection_pro.insert_many(docs)
        print(f"Insertados {len(docs)} nuevos documentos")
    else:
        print("No hay datos nuevos")