"""Script de ajuste fino por fases del modelo Aurora.

Este punto de entrada ejecuta el entrenamiento configurado en YAML, recorriendo
las distintas fases y learning rates definidos para el experimento final del
TFG y reutilizando checkpoints cuando corresponde.
"""

import gc
import glob
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

from src.config import MDB, PARAMS
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.models.visualizer import AuroraVisualizerCallback


class AuroraTrainingOrchestrator:
    """Orquesta el ajuste fino de Aurora a través de fases y learning rates."""

    def get_best_checkpoint_from_dir(self, dir_path, force_best=False):
        """Localiza el checkpoint a reutilizar según la política actual."""

        last_path = os.path.join(dir_path, "last.ckpt")

        if force_best:
            ckpts = glob.glob(os.path.join(dir_path, "*.ckpt"))
            ckpts = [checkpoint for checkpoint in ckpts if "last" not in checkpoint]
            if ckpts:
                try:
                    return min(
                        ckpts,
                        key=lambda path: float(
                            os.path.basename(path).split("-")[-1].replace(".ckpt", "")
                        ),
                    )
                except Exception:
                    ckpts.sort(key=os.path.getmtime)
                    return ckpts[-1]

        if os.path.exists(last_path):
            return last_path

        ckpts = glob.glob(os.path.join(dir_path, "*.ckpt"))
        if ckpts:
            ckpts.sort(key=os.path.getmtime)
            return ckpts[-1]

        return None

    def run(self):
        """Ejecuta el entrenamiento completo siguiendo la configuración activa."""

        torch.set_float32_matmul_precision("medium")

        mejor_checkpoint_global = PARAMS["aurora_base"].get("checkpoint")
        if mejor_checkpoint_global and not os.path.exists(mejor_checkpoint_global):
            mejor_checkpoint_global = None

        for nombre_fase, conf_fase in PARAMS["fases"].items():
            print(f"\n{'=' * 60}\n>>> INICIANDO: {nombre_fase.upper()}\n{'=' * 60}")
            mejor_loss_de_fase = float("inf")
            mejor_ckpt_de_fase = None

            for lr in conf_fase["learning_rates"]:
                lr_str = f"{lr:.0e}".replace("e-0", "e-")
                ckpt_dir = os.path.join("checkpoints", nombre_fase, f"lr_{lr_str}")
                os.makedirs(ckpt_dir, exist_ok=True)

                done_flag = os.path.join(ckpt_dir, "completed.txt")
                if os.path.exists(done_flag):
                    print(f"[SALTANDO]: {nombre_fase} | LR: {lr_str} ya completado.")
                    best_in_dir = self.get_best_checkpoint_from_dir(ckpt_dir, force_best=True)
                    if best_in_dir:
                        mejor_checkpoint_global = best_in_dir
                        mejor_ckpt_de_fase = best_in_dir
                    continue

                print(f"\n[Ejecutando]: {nombre_fase} | LR: {lr_str} | Forecast: {conf_fase['forecast_hours']}h")

                cfg_iteracion = PARAMS["aurora_base"].copy()
                cfg_iteracion.update(
                    {
                        "learning_rate": lr,
                        "epochs": conf_fase["epochs"],
                        "target_hours": conf_fase["target_hours"],
                        "forecast_hours": conf_fase["forecast_hours"],
                        "min_delta": conf_fase["min_delta"],
                    }
                )

                dm = AuroraDataModule(cfg_mdb=MDB, cfg_aurora=cfg_iteracion)
                dm.setup()

                path_to_load = self.get_best_checkpoint_from_dir(
                    ckpt_dir, force_best=False
                ) or mejor_checkpoint_global

                if path_to_load:
                    print(f"--- Cargando pesos desde: {path_to_load} ---")
                    model = AuroraFinetuner.load_from_checkpoint(
                        path_to_load,
                        cfg_coords={"lats": dm.lats, "lons": dm.lons},
                        cfg_aurora=cfg_iteracion,
                        strict=False,
                    )
                else:
                    model = AuroraFinetuner(
                        cfg_coords={"lats": dm.lats, "lons": dm.lons},
                        cfg_aurora=cfg_iteracion,
                    )

                checkpoint_callback = ModelCheckpoint(
                    monitor="val/loss",
                    dirpath=ckpt_dir,
                    filename="aurora-{epoch:02d}-{val/loss:.4f}",
                    auto_insert_metric_name=False,
                    save_top_k=1,
                    mode="min",
                    save_last=True,
                )

                trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=1,
                    precision="bf16-mixed",
                    max_epochs=cfg_iteracion["epochs"],
                    accumulate_grad_batches=192,
                    gradient_clip_val=0.5,
                    callbacks=[
                        checkpoint_callback,
                        EarlyStopping(
                            monitor="val/loss",
                            patience=15,
                            min_delta=cfg_iteracion["min_delta"],
                            mode="min",
                        ),
                        AuroraVisualizerCallback(
                            base_dir=os.path.join(
                                "docs/entrenamiento",
                                nombre_fase,
                                f"lr_{lr_str}",
                            )
                        ),
                    ],
                    log_every_n_steps=10,
                )

                try:
                    trainer.fit(model, datamodule=dm)

                    with open(done_flag, "w", encoding="utf-8") as f:
                        f.write(
                            "Status: COMPLETED\n"
                            f"Timestamp: {datetime.now()}\n"
                            f"Best Loss: {checkpoint_callback.best_model_score:.4f}\n"
                        )

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


def get_best_checkpoint_from_dir(dir_path, force_best=False):
    """Mantiene la API histórica basada en función para resolver checkpoints."""

    return AuroraTrainingOrchestrator().get_best_checkpoint_from_dir(dir_path, force_best=force_best)


def run_orchestrator():
    """Mantiene la API histórica basada en función para lanzar el entrenamiento."""

    return AuroraTrainingOrchestrator().run()


if __name__ == "__main__":
    run_orchestrator()
