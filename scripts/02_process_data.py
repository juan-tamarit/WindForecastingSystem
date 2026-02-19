from src.frame.DFmanager import DFmanager
import logging

df_manager=DFmanager()

if __name__ == "__main__":
    logging.info("Construyendo datosProcesados...")
    
    # 1. Fecha última procesada
    if df_manager.getCollectionPro().count_documents({}) > 0:
        pipeline = [{"$sort": {"valid_time": -1}}, {"$limit": 1}]
        last_doc = list(df_manager.getCollectionPro().aggregate(pipeline))[0]
        after_date = last_doc["valid_time"]
    else:
        after_date = None
        logging.info("Primera carga")
    
    # 2. Solo datos nuevos desde Mongo (filtro eficiente)
    df_raw_new = df_manager.getDataFrame(after_date)
    logging.info(f"Datos raw nuevos: {df_raw_new.shape}")
    
    if len(df_raw_new) == 0:
        print("No hay datos nuevos")
    else:
        # 3. Procesar solo nuevos
        df_proc_new = df_manager.addFeatures(df_raw_new)
        docs = df_proc_new.to_dict(orient="records")
        df_manager.getCollectionPro().insert_many(docs)
        print(f"Insertados {len(docs)} documentos nuevos")