import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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

def buildTFTDataSet(df,targets,static_reals,time_varying_known_reals,time_varying_unknown_reals,max_encoder_length,max_prediction_length):
    min_len = max_encoder_length + max_prediction_length

    valid_locations = (df.groupby("location_id").size().loc[lambda x: x >= min_len].index)
    df = df[df["location_id"].isin(valid_locations)]

    training=TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=targets,
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
    return training

# -----------------------------------------------------------------------------
# Creación del TimeSeriesDataSet de validación a partir del de entrenamiento
#
# Objetivo:
# - Generar el dataset de validación reutilizando toda la configuración
#   del TimeSeriesDataSet de entrenamiento (escalados, encoders, etc.).
#
# Entrada:
# - training: TimeSeriesDataSet ya construido sobre df_train.
#             Contiene la definición completa del problema:
#             targets, group_ids, longitudes de ventana, escalados, etc.
# - df_val: DataFrame con la parte final del histórico (zona de validación),
#           con las mismas columnas y esquema que df_train.
#
# Implementación:
# - TimeSeriesDataSet.from_dataset(training, df_val, ...)
#   · Copia la configuración del dataset de entrenamiento.
#   · Recalcula solo los índices / ventanas sobre df_val. [web:354][web:358]
# - min_prediction_idx:
#   · Se fija en el mínimo time_idx de df_val para garantizar que
#     la validación se hace exclusivamente sobre el bloque temporal de test.
# - stop_randomization=True:
#   · Desactiva cualquier aleatorización interna en la generación de ejemplos,
#     haciendo la evaluación determinista y más interpretable. [web:358]
#
# Resultado:
# - Un TimeSeriesDataSet de validación perfectamente alineado con training,
#   listo para usarse en el val_dataloader del Trainer.
# -----------------------------------------------------------------------------

def buildValidation(training,df_val):
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_val,
        min_prediction_idx=df_val["time_idx"].min(),
        stop_randomization=True,
    )
    return validation

# -------------------------------------------------------------
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

def buildTFTModel(training):
    tft= TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=48,
    attention_head_size=4,
    hidden_continuous_size=24,
    dropout=0.15,
    learning_rate=1e-3,
    loss=QuantileLoss(),
    log_interval=-1,        # no log de interpretación en train
    log_val_interval=-1,    # no log de interpretación en val
    reduce_on_plateau_patience=4,
    )
    return tft

# ---------------------------------------------
# Entrenamiento del modelo TFT con PyTorch Lightning
#
# 1) DataLoaders:
# - train_dataloader y val_dataloader convierten los TimeSeriesDataSet
#   de entrenamiento y validación en DataLoaders de PyTorch.
# - Generan batches de ventanas temporales (encoder + decoder) a partir
#   de time_idx y location_id, listas para el TFT.
# - batch_size controla cuántas secuencias se procesan en paralelo y
#   num_workers el número de procesos de carga (0 es lo más seguro en Windows). [web:354]
#
# 2) Callbacks:
# - EarlyStopping:
#   · monitor="val_loss": observa la pérdida de validación.
#   · patience=3, min_delta=1e-4: detiene el entrenamiento cuando la mejora
#     en val_loss deja de ser significativa durante varios epochs.
# - LearningRateMonitor:
#   · Registra la evolución del learning rate durante el entrenamiento.
# - ModelCheckpoint:
#   · dirpath="checkpoints": carpeta donde se guardan los checkpoints.
#   · filename="tft-{epoch:02d}-{val_loss:.4f}": nombre con epoch y val_loss.
#   · monitor="val_loss", mode="min": guarda el mejor modelo según val_loss.
#   · save_top_k=1: conserva solo el mejor.
#   · save_last=True: opcionalmente guarda también el último epoch. [web:362]
#
# 3) Logger:
# - TensorBoardLogger("lightning_logs") registra métricas y curvas
#   (loss, val_loss, learning rate, etc.) para visualizarlas con TensorBoard. [web:300]
#
# 4) Trainer:
# - max_epochs: número máximo de epochs (el early stopping puede cortar antes).
# - accelerator="auto": selecciona CPU o GPU según disponibilidad.
# - gradient_clip_val=0.1: recorta gradientes para evitar explosiones. [web:371]
# - callbacks=[...]: integra EarlyStopping, LearningRateMonitor y ModelCheckpoint.
# - logger=logger: activa el logging de métricas durante train y val.
#
# Ventajas:
# - Lightning gestiona automáticamente el bucle de entrenamiento,
#   validación, logging y guardado del mejor modelo.
# - No es necesario escribir a mano la lógica de epochs, backprop
#   ni el guardado de checkpoints; se controla todo desde Trainer +
#   callbacks manteniendo el código del modelo limpio.
# ---------------------------------------------

def trainTFT(training,validation,tft,batch_size,max_epochs):
    train_dataloader=training.to_dataloader(
    train=True,
    batch_size=batch_size,
    num_workers=10,
    persistent_workers=True
    )

    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=10,
        persistent_workers=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        min_delta=1e-4,
        mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    checkpoint_callback=ModelCheckpoint(
        dirpath="checkpoints",
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger,checkpoint_callback],
        logger=logger
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    return checkpoint_callback

# -------------------------------------------------------------
# Carga del mejor modelo TFT desde un checkpoint
#
# Objetivo:
# - Reconstruir el modelo TemporalFusionTransformer tal y como fue
#   guardado durante el entrenamiento (weights, arquitectura y
#   configuración asociada al TimeSeriesDataSet).
#
# Parámetros:
# - checkpoint_path (str):
#     Ruta completa al archivo .ckpt que contiene el mejor modelo,
#     típicamente obtenido con:
#       best_checkpoint_path = checkpoint_callback.best_model_path
#
# Comportamiento:
# - TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
#     Carga los pesos y la configuración del modelo desde el fichero
#     de checkpoint.
# - model.eval()
#     Pone el modelo en modo evaluación (no entrenamiento), desactivando
#     comportamientos específicos de train como dropout y evitando que
#     se sigan actualizando estadísticas internas.
#
# Uso típico:
# - Tras entrenar con trainTFT(...), obtener la ruta del mejor modelo:
#       best_checkpoint_path = checkpoint_callback.best_model_path
# - Cargar el modelo listo para inferencia o cálculo de métricas:
#       best_tft = loadBestModel(best_checkpoint_path)
# -------------------------------------------------------------

def loadBestModel(checkpoint_path):
    model=TemporalFusionTransformer.load_from_checkpoint(checkpoint_path,weights_only=False) # weights_only=False permite explícitamente la clase MultiNormalizer
    model.eval()
    return model