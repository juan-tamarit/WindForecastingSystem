#imports

from app.era5 import getDataERA5 
from app.aemet import getDataAemet
from datetime import datetime, timedelta
from app.DBmanager import loadIntoDB,getDataFrame
from app.DFmanager import addFeatures
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor



#funciones
def setDates(fecha_ini_dt,fecha_fin_dt):
    fecha_ini_AEMET=fecha_ini_dt.strftime("%Y-%m-%d")+"T00:00:00UTC"
    fecha_fin_AEMET=fecha_fin_dt.strftime("%Y-%m-%d")+"T23:59:59UTC"
    dates={"aemet":[fecha_ini_AEMET,fecha_fin_AEMET],"era5":[fecha_ini_dt,fecha_fin_dt]}
    return dates
def loadData(start,end):
    current_start=start
    control=1
    while current_start<=end:
        #gestión del bucle
        current_end= current_start + timedelta(days=15)
        if current_end>end:
            current_end=end
        #obtención y carga de los datos
        dates= setDates(current_start,current_end)
        data=getDataERA5(dates["era5"][0],dates["era5"][1])
        loadIntoDB(data,control)
        #siguiente fecha
        current_start=current_end+timedelta(days=1)
#variables
start=datetime(2024,1,1)
end=datetime(2024,1,16)
#código
#loadData(start,end)
df=getDataFrame()
df=addFeatures(df)

targets = ["wind_speed", "wind_dir"]
static_reals = ["latitude", "longitude", "elevacion_m"]
time_varying_known_reals = ["time_idx"]
time_varying_unknown_reals = ["u10", "v10", "t2m", "d2m", "msl", "sp","tcwv", "cape", "blh","wind_speed", "wind_dir"]

max_encoder_length=48
max_prediction_length=6

# -----------------------------------------------------------------------------
# Creación del TimeSeriesDataSet (PyTorch Forecasting - TFT)
#
# Este objeto define formalmente el problema de predicción en series temporales.
# Se encarga automáticamente de:
#  - Construir las ventanas temporales (encoder / decoder)
#  - Separar pasado y futuro
#  - Normalizar internamente los targets por serie
#  - Gestionar múltiples series temporales (una por location_id)
#
# Parámetros principales:
# - df: DataFrame ordenado temporalmente y enriquecido con features
# - time_idx: índice temporal entero y regular (en horas)
# - target: variables objetivo a predecir (multivariable)
# - group_ids: identificador de cada serie temporal independiente (lat, lon)
#
# Ventanas temporales:
# - max_encoder_length: número de pasos pasados usados como contexto (48 h)
# - max_prediction_length: horizonte de predicción futura (6 h)
#
# Tipos de variables:
# - static_reals: variables fijas por serie (no varían en el tiempo)
# - time_varying_known_reals: variables futuras conocidas (ej. time_idx)
# - time_varying_unknown_reals: variables observadas solo hasta el presente,
#   incluyendo las propias targets en el pasado
#
# Opciones adicionales:
# - add_relative_time_idx: añade un índice temporal relativo dentro de la ventana
# - add_target_scales: normalización automática por serie
# - add_encoder_length: informa al modelo de la longitud real del encoder
# -----------------------------------------------------------------------------
training_speed=TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target=targets[0],
    group_ids=["location_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_reals=static_reals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True  # <--- esto permite saltos en los time_idx
)
training_dir=TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target=targets[1],
    group_ids=["location_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_reals=static_reals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True  # <--- esto permite saltos en los time_idx
)
# -------------------------------------------------------------
# Definición del DataLoader y del modelo Temporal Fusion Transformer (TFT)
#
# train_dataloader:
#   - Convierte el TimeSeriesDataSet en un DataLoader de PyTorch.
#   - Se encarga de generar los batches de entrenamiento.
#   - Cada batch contiene ventanas temporales creadas internamente por TFT
#     (encoder + decoder) a partir de time_idx y location_id.
#   - batch_size define cuántas secuencias se procesan en paralelo.
#   - num_workers controla los procesos para la carga de datos (0 = seguro en Windows).
#
# TemporalFusionTransformer.from_dataset:
#   - Inicializa el modelo TFT utilizando la estructura del dataset.
#   - El modelo infiere automáticamente:
#       * número de variables
#       * escalado de los datos
#       * dimensiones internas necesarias
#   - Parámetros principales:
#       * learning_rate: tasa de aprendizaje del optimizador.
#       * hidden_size: tamaño de las capas ocultas del modelo.
#       * attention_head_size: número de cabezas de atención temporal.
#       * dropout: regularización para evitar overfitting.
#       * hidden_continuous_size: tamaño de embeddings para variables continuas.
#       * loss: función de pérdida (QuantileLoss para regresión probabilística).
#       * log_interval: frecuencia de logging durante el entrenamiento.
#       * reduce_on_plateau_patience: epochs sin mejora antes de reducir el learning rate.
# -------------------------------------------------------------

train_dataloader=training.to_dataloade(
    train=True,
    batch_size=64,
    num_workers=0
)
tft_speed = TemporalFusionTransformer.from_dataset(
    training_speed,
    learning_rate=1e-3,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)
tft_dir = TemporalFusionTransformer.from_dataset(
    training_dir,
    learning_rate=1e-3,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)
# ---------------------------------------------
# En este bloque se conecta el DataLoader con el modelo Temporal Fusion Transformer (TFT)
# utilizando PyTorch Lightning Trainer, que gestiona automáticamente el bucle de entrenamiento,
# la retropropagación, logging, early stopping y optimización.
#
# Parámetros clave:
# - max_epochs=30: número máximo de epochs (el entrenamiento puede terminar antes si se activa early stopping)
# - accelerator="cpu"/"gpu": define dónde se entrena (CPU o GPU)
# - gradient_clip_val=0.1: recorta los gradientes para evitar inestabilidad numérica
# - EarlyStopping: detiene el entrenamiento si la loss no mejora tras cierto número de epochs
# - LearningRateMonitor: registra cambios en el learning rate durante el entrenamiento
#
# Ventajas:
# - No es necesario normalizar manualmente ni crear ventanas: TFT gestiona internamente estas operaciones
# - Permite entrenar múltiples series temporales con distintos location_id de forma eficiente
# - La QuantileLoss permite predicción probabilística, muy adecuada para datos meteorológicos
# ---------------------------------------------
early_stop_callback = EarlyStopping(
    monitor="train_loss",
    patience=5,
    min_delta=1e-4,
    mode="min"
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = Trainer(
    max_epochs=30,
    accelerator="cpu",
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback, lr_monitor],
    log_every_n_steps=10
)

trainer.fit(
    tft_speed,
    train_dataloaders=train_dataloader
)
trainer.fit(
    tft_dir,
    train_dataloaders=train_dataloader
)