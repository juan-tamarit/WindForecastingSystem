from dotenv import load_dotenv
import os
import cdsapi
import yaml
from pathlib import Path
import numpy as np
from datetime import datetime

load_dotenv()

# Credenciales (sensibles, de .env)
AEMET_API_KEY = os.getenv("AEMET_API_KEY")
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = os.getenv("CDS_API_KEY")
cds = cdsapi.Client(url=CDS_URL, key=CDS_KEY)

# MongoDB
MDB = {
    "uri": "mongodb://localhost:27017/",
    "db_name": "TFG",
    "collection_era": "datosERA5",
    "collection_z": "orografia"
}

# Carga parámetros de YAML
CONFIG_PATH = Path("configs/config.yaml")
PARAMS = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        PARAMS = yaml.safe_load(f)
    
    # Generación automática de grids
    if "data" in PARAMS and "grid" in PARAMS["data"]:
        def make_grid(start, stop, step):
            # +step/2 para incluir stop aproximado (como np.arange original)
            return np.arange(start, stop + step/2, step).tolist()
        
        lats_grid = PARAMS["data"]["grid"]["lats"]
        PARAMS["data"]["lats"] = make_grid(
            lats_grid["start"], lats_grid["stop"], lats_grid["step"]
        )
        
        lons_grid = PARAMS["data"]["grid"]["lons"]
        PARAMS["data"]["lons"] = make_grid(
            lons_grid["start"], lons_grid["stop"], lons_grid["step"]
        )
        
        # Opcional: eliminar grid crudo
        # del PARAMS["data"]["grid"]
    
    # Conversión datetime strings a objetos
    if "data" in PARAMS:
        PARAMS["data"]["start_dt"] = datetime.fromisoformat(PARAMS["data"]["start"])
        PARAMS["data"]["end_dt"] = datetime.fromisoformat(PARAMS["data"]["end"])
else:
    raise FileNotFoundError(f"Crear configs/config.yaml en {CONFIG_PATH}")