from dotenv import load_dotenv
import os
import cdsapi
import yaml
from pathlib import Path
import numpy as np
from datetime import datetime

load_dotenv()


AEMET_API_KEY = os.getenv("AEMET_API_KEY")
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = os.getenv("CDS_API_KEY")
cds = cdsapi.Client(url=CDS_URL, key=CDS_KEY)


MDB = {
    "uri": "mongodb://localhost:27017/",
    "db_name": "TFG",
    "collection_era": "datosERA5",
    "collection_z": "orografia",
    "collection_pro": "datosProcesados"
}


CONFIG_PATH = Path("configs/config.yaml")
PARAMS = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        PARAMS = yaml.safe_load(f)
    
    
    if "data" in PARAMS and "grid" in PARAMS["data"]:
        def make_grid(start, stop, step):
            
            return np.arange(start, stop + step/2, step).tolist()
        
        lats_grid = PARAMS["data"]["grid"]["lats"]
        PARAMS["data"]["lats"] = make_grid(
            lats_grid["start"], lats_grid["stop"], lats_grid["step"]
        )
        
        lons_grid = PARAMS["data"]["grid"]["lons"]
        PARAMS["data"]["lons"] = make_grid(
            lons_grid["start"], lons_grid["stop"], lons_grid["step"]
        )
        if "aurora" in PARAMS:
            PARAMS["aurora"]["lats"] = PARAMS["data"]["lats"]
            PARAMS["aurora"]["lons"] = PARAMS["data"]["lons"]
        
    
    
    if "data" in PARAMS:
        PARAMS["data"]["start_dt"] = datetime.fromisoformat(PARAMS["data"]["start"])
        PARAMS["data"]["end_dt"] = datetime.fromisoformat(PARAMS["data"]["end"])
else:
    raise FileNotFoundError(f"Crear configs/config.yaml en {CONFIG_PATH}")

NORMALIZATION_LIMITS = {
    "2t":  {"min": 265.0, "max": 295.0},   # Kelvin (~ -8°C a 22°C)
    "10u": {"min": -8.0,  "max": 14.0},    # m/s
    "10v": {"min": -11.0, "max": 17.0},    # m/s
    "msl": {"min": 100400.0, "max": 102800.0} # Pa
}