import pytorch_lightning as pl
import torch
from src.config import MDB, PARAMS
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.models.visualizer import AuroraVisualizerCallback
import os

def main():
    torch.set_float32_matmul_precision('medium')
    last_ckpt = "checkpoints/last.ckpt"
    path_to_load = last_ckpt if os.path.exists(last_ckpt) else None

    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints/",
        filename="aurora-ERA5-{epoch:02d}-{val/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="min",
        save_last=True
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

    
    if path_to_load:
        print(f"--- Resumiendo entrenamiento desde {last_ckpt} ---")
    else:
        print("--- Iniciando entrenamiento desde cero (pesos Aurora base) ---")

    trainer.fit(model, datamodule=dm, ckpt_path=path_to_load)

if __name__ == "__main__":
    main()