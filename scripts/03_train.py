"""03_train.py - Entrena modelo TFT."""
import logging
import warnings
from src.config import PARAMS
from src.frame.DFmanager import getProcessedDataFrame, splitDataFrame
from src.models.tft_model import (
    buildTFTDataSet, buildValidation, buildTFTModel, trainTFT, loadBestModel
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # sube de scripts/ a raíz
BEST_CKPT_PATH_FILE = PROJECT_ROOT / "src" / "models" / "best_checkpoint_path.txt"

warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Entrenando modelo TFT")
    
    # Carga y preprocess
    df = getProcessedDataFrame()
    logger.info(f"DataFrame preparado: {df.shape}")
    
    # Config de YAML
    cfg = PARAMS["model"]
    targets = cfg["targets"]
    static_reals = ["latitude", "longitude", "elevacion_m"]
    time_varying_known_reals = ["time_idx"]
    time_varying_unknown_reals = [
        "wind_speed", "wind_dir_sin", "wind_dir_cos", "u10", "v10", "u100", "v100",
        "t2m", "d2m", "skt", "sp", "msl", "tp", "ssrd", "strd"
    ]
    
    train_fact = 0.8
    df_train, df_val = splitDataFrame(df, train_fact)
    
    # Dataset y modelo
    training = buildTFTDataSet(
        df_train, targets, static_reals, time_varying_known_reals,
        time_varying_unknown_reals, cfg["max_encoder_length"], cfg["max_prediction_length"]
    )
    validation = buildValidation(training, df_val)
    tft = buildTFTModel(training)
    
    # Entrenamiento
    batch_size = cfg["batch_size"]
    max_epochs = cfg["max_epochs"]
    checkpoint_callback = trainTFT(training, validation, tft, batch_size, max_epochs)
    BEST_CKPT_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)  # por si acaso
    BEST_CKPT_PATH_FILE.write_text(checkpoint_callback.best_model_path)
    logger.info(f"Mejor checkpoint guardado en: {BEST_CKPT_PATH_FILE}")
    print("Entrenamiento completado.")