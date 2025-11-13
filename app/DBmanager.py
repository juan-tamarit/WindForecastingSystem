from pymongo import MongoClient
from config import mdb
from datetime import datetime

client=MongoClient(mdb["uri"])
db=client[mdb["db_name"]]
coleccion=db[mdb["collection_name"]]
collection_aemet=db[mdb["collection_aemet"]]

def loadIntoDB(data,control):
    try:
        if control==0:
            for doc in data:
                doc['fecha']=datetime.strptime(doc['fecha'],"%Y-%m-%d")
                collection_aemet.insert_one(doc)
        elif control==1:
            coleccion.insert_many(data)
    except Exception as e:
        print(f"Error en la carga: {e}")
        raise