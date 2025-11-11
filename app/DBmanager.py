from pymongo import MongoClient
from config import mdb

client=MongoClient(mdb["uri"])
db=client[mdb["db_name"]]
coleccion=db[mdb["collection_name"]]

def loadIntoDB(data):
    try:
        coleccion.insert_many(data)
    except Exception as e:
        print(f"Error en la carga: {e}")
        raise