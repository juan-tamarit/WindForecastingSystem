from pymongo import MongoClient
from src.config import MDB
from datetime import datetime
import threading
from queue import Queue
import time

class DBmanager:
    def __init__(self):
        self.client = MongoClient(
            MDB["uri"],
            maxPoolSize=50,
            retryWrites=True
        )
        self.db=self.client[MDB["db_name"]]
        self.collection_era=self.db[MDB["collection_era"]]
        self.collection_z=self.db[MDB["collection_z"]]
        self.collection_pro=self.db[MDB["collection_pro"]]
        self._bulk_queue = Queue(maxsize=1000)  # Buffer bulk inserts
        self._bulk_thread = threading.Thread(target=self._bulk_worker, daemon=True)
        self._bulk_thread.start()

    def _bulk_worker(self):
        while True:
            batch = []
            start_time = time.time()
        
            while (time.time() - start_time < 5 and len(batch) < 1000 and not self._bulk_queue.empty()):
                try:
                    batch.append(self._bulk_queue.get(timeout=0.5))
                except:
                    break
        
            if batch:  # Works with single docs too
                try:
                    self.collection_era.insert_many(batch, ordered=False)
                except Exception as e:
                    print(f"Bulk insert error: {e}")
                    for doc in batch:
                        try:
                            self.collection_era.insert_one(doc)
                        except:
                            pass

    def loadIntoDB(self,data, control:int=1):
        if not data:
            return
        try:
            static_dict = self.loadStaticDictMongo()
            processed_data=[]
            for doc in data:
                doc=doc.copy()
                doc['valid_time'] = datetime.strptime(doc['valid_time'], "%Y-%m-%d %H:%M:%S")
                if control==1 and static_dict:
                    lat = round(float(doc['latitude']), 2)
                    lon = round(float(doc['longitude']), 2)
                    vals = static_dict.get((lat, lon))
                    if vals:
                        doc['z'] = vals["z"]
                        doc['lsm'] = vals["lsm"]
                processed_data.append(doc)        
            for doc in processed_data:
                try:
                    self._bulk_queue.put_nowait(doc)
                except:
                    self.collection_era.insert_one(doc)
        except Exception as e:
            print(f"Error en la carga: {e}")
            raise

    def saveStaticDictMongo(self,static_dict:dict):
        docs = []
        for (lat, lon), vals in static_dict.items():
            docs.append({
                "latitude": lat,
                "longitude": lon,
                "z": vals["z"],
                "lsm": vals["lsm"]
            })
        if docs:
            self.collection_z.insert_many(docs)

    def loadStaticDictMongo(self)->dict:
        static_dict = {}
        pipeline = [{"$project": {"_id": 0, "latitude": 1, "longitude": 1, "z": 1, "lsm": 1}}]
        for doc in self.collection_z.aggregate(pipeline):
            lat = float(doc["latitude"])
            lon = float(doc["longitude"])
            static_dict[(lat, lon)] = {
                "z": float(doc["z"]),
                "lsm": float(doc["lsm"])
            }
        return static_dict
    
    def get_last_date_for_point(self, lat: float, lon: float):
        lat_r = round(lat, 2)
        lon_r = round(lon, 2)
    
        pipeline = [
            {"$match": {"latitude": lat_r, "longitude": lon_r}},
            {"$sort": {"valid_time": -1}},
            {"$limit": 1},
            {"$project": {"valid_time": 1}}
        ]
    
        result = list(self.collection_era.aggregate(pipeline))
        return result[0]["valid_time"] if result else None

