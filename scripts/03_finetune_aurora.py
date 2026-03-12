import os
import gc
import torch
import glob
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.config import MDB, PARAMS
from datetime import datetime
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.models.visualizer import AuroraVisualizerCallback

def get_best_checkpoint_from_dir(dir_path, force_best=False):
    """
    force_best=True: Busca el .ckpt con menor loss (para herencia entre etapas).
    force_best=False: Prioriza 'last' para reanudar sesiones en la misma carpeta.
    """
    last_path = os.path.join(dir_path, "last.ckpt")
    
    if force_best:
        ckpts = glob.glob(os.path.join(dir_path, "*.ckpt"))
        ckpts = [c for c in ckpts if "last" not in c]
        if ckpts:
            try:
                # Busca el valor numérico después del último '-' (formato: aurora-epoch=01-0.2915.ckpt)
                return min(ckpts, key=lambda x: float(os.path.basename(x).split('-')[-1].replace('.ckpt', '')))
            except:
                ckpts.sort(key=os.path.getmtime)
                return ckpts[-1]

    if os.path.exists(last_path):
        return last_path
    
    ckpts = glob.glob(os.path.join(dir_path, "*.ckpt"))
    if ckpts:
        ckpts.sort(key=os.path.getmtime)
        return ckpts[-1]
        
    return None

def run_orchestrator():
    torch.set_float32_matmul_precision('medium')
    
    mejor_checkpoint_global = PARAMS["aurora_base"].get("checkpoint")
    if mejor_checkpoint_global and not os.path.exists(mejor_checkpoint_global):
        mejor_checkpoint_global = None

    for nombre_fase, conf_fase in PARAMS["fases"].items():
        print(f"\n{'='*60}\n>>> INICIANDO: {nombre_fase.upper()}\n{'='*60}")
        mejor_loss_de_fase = float('inf')
        mejor_ckpt_de_fase = None

        for lr in conf_fase["learning_rates"]:
            lr_str = f"{lr:.0e}".replace("e-0", "e-")
            ckpt_dir = os.path.join("checkpoints", nombre_fase, f"lr_{lr_str}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            done_flag = os.path.join(ckpt_dir, "completed.txt")
            if os.path.exists(done_flag):
                print(f"[SALTANDO]: {nombre_fase} | LR: {lr_str} ya completado.")
                best_in_dir = get_best_checkpoint_from_dir(ckpt_dir, force_best=True)
                if best_in_dir:
                    mejor_checkpoint_global = best_in_dir 
                    mejor_ckpt_de_fase = best_in_dir
                continue 

            print(f"\n[Ejecutando]: {nombre_fase} | LR: {lr_str} | Forecast: {conf_fase['forecast_hours']}h")

            cfg_iteracion = PARAMS["aurora_base"].copy()
            cfg_iteracion.update({
                "learning_rate": lr,
                "epochs": conf_fase["epochs"],
                "target_hours": conf_fase["target_hours"],
                "forecast_hours": conf_fase["forecast_hours"]
            })

            dm = AuroraDataModule(cfg_mdb=MDB, cfg_aurora=cfg_iteracion)
            dm.setup()

            # REQUISITO: Prioridad last en local, si no, mejor global de etapa anterior
            path_to_load = get_best_checkpoint_from_dir(ckpt_dir, force_best=False) or mejor_checkpoint_global

            if path_to_load:
                print(f"--- Cargando pesos desde: {path_to_load} ---")
                model = AuroraFinetuner.load_from_checkpoint(
                    path_to_load,
                    cfg_coords={"lats": dm.lats, "lons": dm.lons},
                    cfg_aurora=cfg_iteracion,
                    strict=False
                )
            else:
                model = AuroraFinetuner(cfg_coords={"lats": dm.lats, "lons": dm.lons}, cfg_aurora=cfg_iteracion, stats=stats)

            checkpoint_callback = ModelCheckpoint(
                monitor="val/loss",
                dirpath=ckpt_dir,
                filename="aurora-{epoch:02d}-{val/loss:.4f}",
                auto_insert_metric_name=False,
                save_top_k=1,
                mode="min",
                save_last=True
            )
            
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                precision="bf16-mixed",
                max_epochs=cfg_iteracion["epochs"],
                accumulate_grad_batches=192,
                gradient_clip_val=0.5,
                callbacks=[checkpoint_callback, EarlyStopping(monitor="val/loss", patience=15,min_delta=0.0001,mode="min"), AuroraVisualizerCallback(base_dir=os.path.join("docs/entrenamiento", nombre_fase, f"lr_{lr_str}"))],
                log_every_n_steps=10
            )

            try:
                trainer.fit(model, datamodule=dm)
                
                with open(done_flag, "w", encoding='utf-8') as f:
                    f.write(f"Status: COMPLETED\nTimestamp: {datetime.now()}\nBest Loss: {checkpoint_callback.best_model_score:.4f}\n")

                # REQUISITO: Al terminar con éxito, este se vuelve el 'mejor global' para el siguiente paso
                if checkpoint_callback.best_model_path:
                    mejor_checkpoint_global = checkpoint_callback.best_model_path
                    if checkpoint_callback.best_model_score < mejor_loss_de_fase:
                        mejor_loss_de_fase = checkpoint_callback.best_model_score
                        mejor_ckpt_de_fase = checkpoint_callback.best_model_path

            except KeyboardInterrupt:
                print("\n[PAUSA] Interrumpido por usuario.")
                sys.exit(0)

            del model, trainer, dm
            gc.collect()
            torch.cuda.empty_cache()

        if mejor_ckpt_de_fase:
            mejor_checkpoint_global = mejor_ckpt_de_fase
            print(f"\n>>> GANADOR FASE {nombre_fase.upper()}: {mejor_checkpoint_global}")

if __name__ == "__main__":
    run_orchestrator()