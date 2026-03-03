import pytorch_lightning as pl
import torch
from src.config import MDB, PARAMS
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping # Importamos EarlyStopping
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.models.visualizer import AuroraVisualizerCallback
import os

def main():
    torch.set_float32_matmul_precision('medium')
    last_ckpt = PARAMS["auora"].get("checkpoint")
    path_to_load = last_ckpt if os.path.exists(last_ckpt) else None

    # 1. Configuración del Checkpoint (Ya lo tenías, mantenemos el mejor)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints/",
        filename="aurora-ERA5-{epoch:02d}-{val/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="min",
        save_last=True
    )

    # 2. Configuración del Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val/loss",      # Vigila la pérdida de validación
        patience=15,             # Si no mejora en 15 épocas, se detiene
        verbose=True,            # Te avisa en consola cuando para
        mode="min",              # Buscamos minimizar la pérdida
        min_delta=0.0001         # Mejora mínima para considerarse avance
    )

    dm = AuroraDataModule(
        cfg_mdb=MDB, 
        cfg_aurora=PARAMS["aurora"]
    )
    
    dm.setup()

    model = AuroraFinetuner(
        cfg_coords={
            "lats": dm.lats,
            "lons": dm.lons
        },
        cfg_aurora=PARAMS["aurora"]
    )

    visualizer = AuroraVisualizerCallback(base_dir="docs/entrenamiento")

    # 3. Trainer actualizado con los nuevos parámetros
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=100,          # Subimos a 100 como sugirieron, EarlyStopping cortará antes
        accumulate_grad_batches=192,
        gradient_clip_val=0.5,
        # Añadimos early_stop_callback a la lista
        callbacks=[checkpoint_callback, early_stop_callback, visualizer],
        log_every_n_steps=10
    )

    if path_to_load:
        print(f"--- Resumiendo entrenamiento desde {last_ckpt} ---")
    else:
        print("--- Iniciando entrenamiento desde cero (pesos Aurora base) ---")

    trainer.fit(model, datamodule=dm, ckpt_path=path_to_load)

if __name__ == "__main__":
    main()