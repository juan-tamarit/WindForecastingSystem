from dotenv import load_dotenv
import os
import cdsapi
#localización del .env
load_dotenv()
#carga de keys
#AEMET
api_key=os.getenv("AEMET_API_KEY")
#CDS
cds_url=f"https://cds.climate.copernicus.eu/api"
cds_key=os.getenv("CDS_API_KEY")
cds=cdsapi.Client(url=cds_url, key=cds_key)
#Mongo
mdb={"uri": "mongodb://localhost:27017/","db_name":"TFG","collection_name":"datosTFG"}