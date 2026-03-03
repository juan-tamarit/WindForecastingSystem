import pytorch_lightning as pl
import torch
from src.config import MDB, PARAMS
from src.frame.DFmanager import DFmanager
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping # Importamos EarlyStopping
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.models.visualizer import AuroraVisualizerCallback
import os

def main():
    torch.set_float32_matmul_precision('medium')
    last_ckpt = PARAMS["aurora"].get("checkpoint")
    path_to_load = last_ckpt if os.path.exists(last_ckpt) else None
    dfm = DFmanager()
    stats = dfm.get_normalization_stats()

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints/",
        filename="aurora-fase1-{epoch:02d}-{val/loss:.4f}", 
        auto_insert_metric_name=False,
        save_top_k=1,          
        mode="min",
        save_last=True,        
        every_n_epochs=1 
    )

    # 2. Configuración del Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val/loss",      
        patience=15,             
        verbose=True,            
        mode="min",              
        min_delta=0.0001         
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
        cfg_aurora=PARAMS["aurora"],
        stats=stats
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