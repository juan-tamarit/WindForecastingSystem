import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
from pathlib import Path

class AuroraVisualizerCallback(Callback):
    def __init__(self, base_dir="docs/entrenamiento"):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # 1. Crear carpeta específica para la época
        epoch = trainer.current_epoch
        epoch_dir = self.base_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # 2. Obtener métricas promedio de la época
        # Lightning las guarda en trainer.callback_metrics
        metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
        
        # 3. Guardar métricas en un CSV
        df = pd.DataFrame([metrics])
        df.to_csv(epoch_dir / "metricas.csv", index=False)

        # 4. Generar y guardar el Mapa PNG
        # Usamos el primer batch de validación disponible para la foto
        # (Nota: Esto requiere que guardes una referencia al batch en el módulo o lo pidas aquí)
        self._save_map_image(trainer, pl_module, epoch_dir)

    def _save_map_image(self, trainer, pl_module, epoch_dir):
        pl_module.eval()
        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader)) 
        
        # ASEGURAR DISPOSITIVO
        # Mueve todo el diccionario de tensores a la GPU (o donde esté el modelo)
        device = pl_module.device
        batch = {
            "inputs": {k: v.to(device) for k, v in batch["inputs"].items()},
            "targets": {
                step: {k: v.to(device) for k, v in vars.items()} 
                for step, vars in batch["targets"].items()
            },
            "statics": {k: v.to(device) for k, v in batch["statics"].items()}
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
            axes[1].set_title(f"Predicción Época {trainer.current_epoch}")
            
            plt.savefig(epoch_dir / "comparativa_viento.png")
            plt.close()