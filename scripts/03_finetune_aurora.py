import pytorch_lightning as pl
import torch
from src.config import MDB, PARAMS  # Importamos PARAMS que ya tiene el YAML procesado
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.models.visualizer import AuroraVisualizerCallback
import os

def main():
    # Optimización para GPUs modernas (RTX 3050+)
    torch.set_float32_matmul_precision('medium')
    last_ckpt = "checkpoints/last.ckpt"
    path_to_load = last_ckpt if os.path.exists(last_ckpt) else None

    # 1. Configurar el Checkpoint usando valores del YAML si fuera necesario, 
    # pero mantenemos la lógica de guardado del mejor modelo.
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints/",
        filename="aurora-ERA5-{epoch:02d}-{val/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="min",
        save_last=True
    )

    # 2. Inicializar el DataModule con los diccionarios de src.config
    # Ahora pasamos el bloque 'aurora' que contiene batch_size, split, etc.
    dm = AuroraDataModule(
        cfg_mdb=MDB, 
        cfg_aurora=PARAMS["aurora"]
    )
    
    # Preparamos los datos para obtener las coordenadas lats/lons reales
    dm.setup()

    # 3. Inicializar el Modelo
    # Le pasamos las coordenadas calculadas y la configuración de Aurora (LR, WD, etc.)
    model = AuroraFinetuner(
        cfg_coords={
            "lats": dm.lats,
            "lons": dm.lons
        },
        cfg_aurora=PARAMS["aurora"]
    )

    # 4. Configurar el Trainer
    # Sacamos max_epochs directamente del YAML
    visualizer = AuroraVisualizerCallback(base_dir="docs/entrenamiento")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=PARAMS["aurora"]["epochs"],
        accumulate_grad_batches=192,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback,visualizer],
        log_every_n_steps=10
    )

    # 5. ¡A entrenar!
    if path_to_load:
        print(f"--- Resumiendo entrenamiento desde {last_ckpt} ---")
    else:
        print("--- Iniciando entrenamiento desde cero (pesos Aurora base) ---")

    trainer.fit(model, datamodule=dm, ckpt_path=path_to_load)

if __name__ == "__main__":
    main()