"""Callbacks de visualización del entrenamiento.

Este módulo define el callback usado durante el ajuste fino para guardar
métricas por época y una comparación visual sencilla entre target y predicción.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning.callbacks import Callback
import torch


class AuroraVisualizerCallback(Callback):
    """Guarda artefactos de seguimiento al final de cada época de validación."""

    def __init__(self, base_dir="docs/entrenamiento"):
        """Prepara el directorio base donde se almacenan las salidas."""

        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Guarda métricas e imagen comparativa al terminar cada validación."""

        epoch = trainer.current_epoch
        epoch_dir = self.base_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}

        df = pd.DataFrame([metrics])
        df.to_csv(epoch_dir / "metricas.csv", index=False)

        self._save_map_image(trainer, pl_module, epoch_dir)

    def _save_map_image(self, trainer, pl_module, epoch_dir):
        """Genera una imagen comparando el target y la predicción del primer paso."""

        pl_module.eval()
        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))

        device = pl_module.device
        batch = {
            "inputs": {k: v.to(device) for k, v in batch["inputs"].items()},
            "targets": {
                step: {k: v.to(device) for k, v in vars.items()}
                for step, vars in batch["targets"].items()
            },
            "statics": {k: v.to(device) for k, v in batch["statics"].items()},
        }

        with torch.no_grad():
            aurora_batch = pl_module.prepare_aurora_batch(batch)
            pred = pl_module(aurora_batch)

            target = batch["targets"]["step_1"]["10u"][0, 0, ..., pl_module.lon_indices].cpu().numpy()
            prediction = pred.surf_vars["10u"][0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(target, origin="lower")
            axes[0].set_title("Realidad ERA5")
            axes[1].imshow(prediction, origin="lower")
            axes[1].set_title(f"PredicciÃ³n Ã‰poca {trainer.current_epoch}")

            plt.savefig(epoch_dir / "comparativa_viento.png")
            plt.close()
