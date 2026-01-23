"""04_evaluate.py - Evalúa modelo TFT con métricas y plots."""
import logging
import warnings
from src.config import PARAMS
from src.frame.DFmanager import getProcessedDataFrame, splitDataFrame
from src.models.tft_model import buildTFTDataSet, buildValidation, loadBestModel
from src.models.metricas import evaluateTarget, plotPredictions, plotErrorHistogram
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_CKPT_PATH_FILE = PROJECT_ROOT / "src" / "models" / "best_checkpoint_path.txt"

warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":    
    logger.info("Evaluando modelo TFT")
    
    # Recupera path del mejor modelo
    best_checkpoint_path = BEST_CKPT_PATH_FILE.read_text().strip()
    best_tft = loadBestModel(best_checkpoint_path)
    logger.info(f"Modelo cargado desde: {best_checkpoint_path}")
    
    # Regenera datos y datasets
    df = getProcessedDataFrame()
    
    print("Columns:", df.columns.tolist())
    print("Dtypes:", df.dtypes[["valid_time", "time_idx", "location_id"]])
    print(df[["wind_speed", "wind_dir_sin", "wind_dir_cos"]].describe())
    
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
    
    training = buildTFTDataSet(
        df_train, targets, static_reals, time_varying_known_reals,
        time_varying_unknown_reals, cfg["max_encoder_length"], cfg["max_prediction_length"]
    )
    validation = buildValidation(training, df_val)
    
    batch_size = cfg["batch_size"]
    
    # Métricas exactas del original
    for i, name in enumerate(targets):
        metrics = evaluateTarget(best_tft, validation, target_idx=i, batch_size=batch_size)
        print(f"== {name} ==")
        print(f"  MAE  : {metrics['MAE']:.4f}")
        print(f"  RMSE : {metrics['RMSE']:.4f}")
        print(f"  MAPE : {metrics['MAPE']:.2f}%")
    
    # Plots exactos del original
    plotPredictions(best_tft, validation, batch_size, 3)
    
    for i, name in enumerate(targets):
        print(f"Mostrando histograma de errores para {name}")
        plotErrorHistogram(best_tft, validation, batch_size, i)
    
    logger.info("Evaluación completada.")