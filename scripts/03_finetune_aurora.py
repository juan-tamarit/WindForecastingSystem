import os
import gc
import torch
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.config import MDB, PARAMS
from src.frame.DFmanager import DFmanager
from datetime import datetime
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.models.visualizer import AuroraVisualizerCallback

def get_best_checkpoint_from_dir(dir_path):
    """
    Busca el archivo .ckpt con el menor val/loss en una carpeta específica.
    Formato esperado: aurora-epoch=XX-val/loss=X.XXXX.ckpt
    """
    ckpts = glob.glob(os.path.join(dir_path, "*.ckpt"))
    ckpts = [c for c in ckpts if "last" not in c]
    if not ckpts:
        return None
    
    try:
        # Extrae el valor numérico después de 'loss=' para comparar
        return min(ckpts, key=lambda x: float(x.split('loss=')[-1].replace('.ckpt', '')))
    except Exception as e:
        print(f"Error parseando checkpoints en {dir_path}: {e}")
        return ckpts[0] # Fallback al primero que encuentre

def run_orchestrator():
    torch.set_float32_matmul_precision('medium')
    dfm = DFmanager()
    stats = dfm.get_normalization_stats()
    
    # 1. Punto de partida global
    mejor_checkpoint_global = PARAMS["aurora_base"].get("checkpoint")
    if mejor_checkpoint_global and not os.path.exists(mejor_checkpoint_global):
        mejor_checkpoint_global = None

    # 2. Bucle de Fases
    for nombre_fase, conf_fase in PARAMS["fases"].items():
        print(f"\n{'='*60}\n>>> INICIANDO: {nombre_fase.upper()}\n{'='*60}")
        
        mejor_loss_de_fase = float('inf')
        mejor_ckpt_de_fase = None

        # 3. Bucle de Learning Rates
        for lr in conf_fase["learning_rates"]:
            lr_str = f"{lr:.0e}".replace("e-0", "e-")
            ckpt_dir = os.path.join("checkpoints", nombre_fase, f"lr_{lr_str}")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # --- COMPROBACIÓN DE SALTO (ROBUSTA) ---
            done_flag = os.path.join(ckpt_dir, "completed.txt")
            if os.path.exists(done_flag):
                print(f"[SALTANDO]: {nombre_fase} | LR: {lr_str} ya fue completado anteriormente.")
                # Actualizamos las variables de la fase para que la siguiente fase tenga el mejor modelo
                best_in_dir = get_best_checkpoint_from_dir(ckpt_dir)
                if best_in_dir:
                    mejor_ckpt_de_fase = best_in_dir
                continue 

            print(f"\n[Ejecutando]: {nombre_fase} | LR: {lr_str} | Forecast: {conf_fase['forecast_hours']}h")

            # Preparar configuración
            cfg_iteracion = PARAMS["aurora_base"].copy()
            cfg_iteracion.update({
                "learning_rate": lr,
                "epochs": conf_fase["epochs"],
                "target_hours": conf_fase["target_hours"],
                "forecast_hours": conf_fase["forecast_hours"]
            })

            # DataModule
            dm = AuroraDataModule(cfg_mdb=MDB, cfg_aurora=cfg_iteracion)
            dm.setup()

            # Decidir carga
            path_to_load = get_best_checkpoint_from_dir(ckpt_dir) or mejor_checkpoint_global

            if path_to_load:
                print(f"--- Cargando pesos desde: {path_to_load} ---")
                model = AuroraFinetuner.load_from_checkpoint(
                    path_to_load,
                    cfg_coords={"lats": dm.lats, "lons": dm.lons},
                    cfg_aurora=cfg_iteracion,
                    stats=stats,
                    strict=False
                )
            else:
                print("--- Iniciando desde pesos Aurora Small base ---")
                model = AuroraFinetuner(
                    cfg_coords={"lats": dm.lats, "lons": dm.lons},
                    cfg_aurora=cfg_iteracion,
                    stats=stats
                )

            # Callbacks
            checkpoint_callback = ModelCheckpoint(
                monitor="val/loss",
                dirpath=ckpt_dir,
                filename="aurora-{epoch:02d}-{val/loss:.4f}",
                auto_insert_metric_name=False,
                save_top_k=1,
                mode="min",
                save_last=True
            )
            
            early_stop = EarlyStopping(
                monitor="val/loss", 
                patience=15, 
                mode="min", 
                min_delta=0.0001
            )
            
            visualizer = AuroraVisualizerCallback(
                base_dir=os.path.join("docs/entrenamiento", nombre_fase, f"lr_{lr_str}")
            )

            # Trainer
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                precision="bf16-mixed",
                max_epochs=cfg_iteracion["epochs"],
                accumulate_grad_batches=192,
                gradient_clip_val=0.5,
                callbacks=[checkpoint_callback, early_stop, visualizer],
                log_every_n_steps=10
            )

            # EJECUTAR ENTRENAMIENTO
            trainer.fit(model, datamodule=dm)

            # --- MARCAR COMO COMPLETADO TRAS EL ÉXITO ---
            with open(done_flag, "w", encoding='utf-8') as f:
                f.write(f"Status: COMPLETED\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Best Loss: {checkpoint_callback.best_model_score:.4f}\n")
                f.write(f"Best CKPT: {checkpoint_callback.best_model_path}\n")

            # Actualizar el mejor de la fase
            current_best_loss = checkpoint_callback.best_model_score
            if current_best_loss is not None and current_best_loss < mejor_loss_de_fase:
                mejor_loss_de_fase = current_best_loss
                mejor_ckpt_de_fase = checkpoint_callback.best_model_path

            # LIMPIEZA
            del model, trainer, dm
            gc.collect()
            torch.cuda.empty_cache()

        # Herencia entre fases
        if mejor_ckpt_de_fase:
            mejor_checkpoint_global = mejor_ckpt_de_fase
            print(f"\n>>> GANADOR FINAL DE {nombre_fase.upper()}: {mejor_checkpoint_global}")
            print(f">>> LOSS: {mejor_loss_de_fase:.4f}")

if __name__ == "__main__":
    run_orchestrator()