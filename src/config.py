"""Configuración central del proyecto.

Este módulo carga las variables de entorno necesarias para acceder a los
servicios externos y expone la configuración declarativa definida en
``configs/config.yaml`` para el resto del pipeline.
"""

from datetime import datetime
from pathlib import Path
import os

import cdsapi
from dotenv import load_dotenv
import numpy as np
import yaml


def make_grid(start, stop, step):
    """Construye una rejilla regular incluyendo el extremo final."""

    return np.arange(start, stop + step / 2, step).tolist()


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
    "collection_pro": "datosProcesados",
}


CONFIG_PATH = Path("configs/config.yaml")
PARAMS = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        PARAMS = yaml.safe_load(f)

    if "data" in PARAMS and "grid" in PARAMS["data"]:
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
