"""Dataset, DataModule y módulo Lightning para Aurora.

Este módulo concentra la preparación de tensores a partir de MongoDB y define
las clases principales utilizadas por el entrenamiento y la evaluación del
modelo Aurora en el proyecto.
"""

from datetime import datetime

from aurora.batch import Batch, Metadata
from aurora.model.aurora import AuroraSmallPretrained
from pymongo import MongoClient
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.frame.DFmanager import DFmanager


class AuroraMongoDataset(Dataset):
    """Construye muestras espacio-temporales de Aurora desde MongoDB."""

    def __init__(self, times, cfg_mdb, lats_torch, lons_torch, cfg_aurora):
        """Inicializa el dataset con tiempos, configuración y rejilla objetivo."""

        self.times = times
        self.cfg_mdb = cfg_mdb
        self.cfg_aurora = cfg_aurora
        self.lats = lats_torch.tolist()
        self.lons = lons_torch.tolist()

        self.client = None
        self.col = None

        self.lat_map = {round(float(lat), 2): i for i, lat in enumerate(self.lats)}
        self.lon_map = {round(float(lon), 2): j for j, lon in enumerate(self.lons)}

        self.static_grid = self._load_static_grid()
        self.stats = self._load_stats()

    def _load_static_grid(self):
        """Carga los campos estáticos usados por todas las muestras."""

        client = MongoClient(self.cfg_mdb["uri"])
        col = client[self.cfg_mdb["db_name"]][self.cfg_mdb["collection_pro"]]

        n_lats, n_lons = len(self.lats), len(self.lons)
        elev = torch.zeros(n_lats, n_lons)
        lsm = torch.zeros(n_lats, n_lons)

        cursor = col.find(
            {"valid_time": self.times[0]},
            {"latitude": 1, "longitude": 1, "elevacion_m": 1, "lsm": 1},
        )
        for doc in cursor:
            i, j = self.lat_map.get(doc["latitude"]), self.lon_map.get(doc["longitude"])
            if i is not None and j is not None:
                elev[i, j] = doc.get("elevacion_m", 0)
                lsm[i, j] = doc.get("lsm", 0)

        client.close()
        return {"elevation": elev, "lsm": lsm}

    def _load_stats(self):
        manager = DFmanager()
        stats = manager.get_spatial_stats(self.times, self.lats, self.lons)
        manager.client.close()

        return stats or {
            "2t": {
                "mean": torch.zeros(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
                "std": torch.ones(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
            },
            "10u": {
                "mean": torch.zeros(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
                "std": torch.ones(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
            },
            "10v": {
                "mean": torch.zeros(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
                "std": torch.ones(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
            },
            "msl": {
                "mean": torch.zeros(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
                "std": torch.ones(len(self.lats), len(self.lons), dtype=torch.float32).numpy(),
            },
        }

    def __len__(self):
        """Devuelve el número de ventanas temporales válidas del dataset."""

        h_hist = self.cfg_aurora["input_hours"]
        h_step = self.cfg_aurora["target_hours"]
        h_total = self.cfg_aurora["forecast_hours"]

        n_steps = max(1, h_total // h_step)

        margin = h_hist + (h_step * n_steps)
        return len(self.times) - margin

    def _get_grid(self, time):
        """Reconstruye una rejilla 2D para un instante concreto."""

        n_lats, n_lons = len(self.lats), len(self.lons)
        grids = {k: torch.zeros(n_lats, n_lons) for k in ["2t", "10u", "10v", "msl"]}

        cursor = self.col.find({"valid_time": time})
        for doc in cursor:
            i, j = self.lat_map.get(doc["latitude"]), self.lon_map.get(doc["longitude"])
            if i is not None and j is not None:
                grids["2t"][i, j] = doc.get("t2m", 0)
                grids["10u"][i, j] = doc.get("u10", 0)
                grids["10v"][i, j] = doc.get("v10", 0)
                grids["msl"][i, j] = doc.get("msl", 0)
        return grids

    def __getitem__(self, idx):
        """Devuelve inputs, targets futuros y campos estáticos para una muestra."""

        if self.client is None:
            self.client = MongoClient(self.cfg_mdb["uri"])
            self.col = self.client[self.cfg_mdb["db_name"]][self.cfg_mdb["collection_pro"]]

        h_hist = self.cfg_aurora["input_hours"]
        h_step = self.cfg_aurora["target_hours"]
        h_total = self.cfg_aurora["forecast_hours"]

        n_steps = max(1, h_total // h_step)

        t_past = self.times[idx]
        t_now = self.times[idx + h_hist]

        grid_past = self._get_grid(t_past)
        grid_now = self._get_grid(t_now)

        inputs = {k: torch.stack([grid_past[k], grid_now[k]], dim=0) for k in grid_past.keys()}

        targets = {}
        for s in range(1, n_steps + 1):
            t_target = self.times[idx + h_hist + (h_step * s)]
            grid_tgt = self._get_grid(t_target)
            targets[f"step_{s}"] = {k: grid_tgt[k].unsqueeze(0) for k in grid_tgt.keys()}

        return {"inputs": inputs, "targets": targets, "statics": self.static_grid}


class AuroraDataModule(pl.LightningDataModule):
    """Configura los conjuntos de entrenamiento, validación y test de Aurora."""

    def __init__(self, cfg_mdb, cfg_aurora):
        """Guarda la configuración necesaria para construir los dataloaders."""

        super().__init__()
        self.cfg_mdb = cfg_mdb
        self.cfg_aurora = cfg_aurora
        self.batch_size = cfg_aurora["batch_size"]
        self.num_workers = cfg_aurora.get("num_workers", 4)
        self.lats = None
        self.lons = None

    def setup(self, stage=None):
        """Carga tiempos y rejillas desde MongoDB y define los splits temporales."""

        client = MongoClient(self.cfg_mdb["uri"])
        col = client[self.cfg_mdb["db_name"]][self.cfg_mdb["collection_pro"]]

        all_times = sorted(col.distinct("valid_time"))
        raw_lats = sorted(col.distinct("latitude"), reverse=True)
        raw_lons = sorted(col.distinct("longitude"))

        self.lats = torch.tensor(raw_lats[:32], dtype=torch.float32)
        self.lons = torch.tensor(raw_lons[:56], dtype=torch.float32)

        client.close()

        n_total = len(all_times)

        test_size = int(n_total * 0.15)
        remaining_size = n_total - test_size

        split_val = self.cfg_aurora["train_split"]
        train_end = int(remaining_size * split_val)

        self.train_times = all_times[:train_end]
        self.val_times = all_times[train_end:remaining_size]
        self.test_times = all_times[remaining_size:]

        self.train_dataset = AuroraMongoDataset(
            self.train_times, self.cfg_mdb, self.lats, self.lons, self.cfg_aurora
        )
        self.val_dataset = AuroraMongoDataset(
            self.val_times, self.cfg_mdb, self.lats, self.lons, self.cfg_aurora
        )
        self.test_dataset = AuroraMongoDataset(
            self.test_times, self.cfg_mdb, self.lats, self.lons, self.cfg_aurora
        )

        print("--- DataModule Configurado ---")
        print(f"Split Train/Val (YAML): {split_val}")
        print(f"Fechas Train: {self.train_times[0].date()} a {self.train_times[-1].date()}")
        print(f"Fechas Val:   {self.val_times[0].date()} a {self.val_times[-1].date()}")
        print(f"Fechas Test:  {self.test_times[0].date()} a {self.test_times[-1].date()}")

    def train_dataloader(self):
        """Construye el dataloader de entrenamiento."""

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Construye el dataloader de validación."""

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Construye el dataloader de test."""

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class AuroraFinetuner(pl.LightningModule):
    """Implementa el ajuste fino autoregresivo del modelo Aurora."""

    def __init__(self, cfg_coords, cfg_aurora):
        """Inicializa el modelo preentrenado y la rejilla de trabajo."""

        super().__init__()
        self.save_hyperparameters(ignore=["stats"])
        self.cfg_aurora = cfg_aurora
        self.model = AuroraSmallPretrained()
        self.model.load_checkpoint()

        lons_raw = cfg_coords["lons"]
        lons_mod = lons_raw % 360
        sorted_vals, sorted_indices = torch.sort(lons_mod)

        self.register_buffer("lon_indices", sorted_indices)
        self.register_buffer("lons_sorted", sorted_vals)
        self.register_buffer("lats", cfg_coords["lats"])

    def prepare_aurora_batch(self, mongo_dict):
        """Convierte un batch de MongoDB al formato esperado por Aurora."""

        device = mongo_dict["inputs"]["2t"].device
        current_indices = self.lon_indices.to(device)

        surf_vars = {
            k: mongo_dict["inputs"][k][..., current_indices]
            for k in ("2t", "10u", "10v", "msl")
        }

        z_reordered = mongo_dict["statics"]["elevation"][..., current_indices]
        lsm_reordered = mongo_dict["statics"]["lsm"][..., current_indices]

        static_vars = {
            "z": z_reordered.mean(dim=0) / 1000.0,
            "lsm": lsm_reordered.mean(dim=0),
        }

        b, t, lat, lon = surf_vars["2t"].shape
        atmos_vars = {
            k: torch.zeros(b, t, 4, lat, lon, device=device)
            for k in ("z", "u", "v", "t", "q")
        }

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=self.lats,
                lon=self.lons_sorted,
                time=(datetime.now(),),
                atmos_levels=(100, 250, 500, 850),
            ),
        )

    def forward(self, batch):
        """Ejecuta una pasada hacia delante del modelo Aurora."""

        return self.model(batch)

    def shared_step(self, mongo_batch, stage):
        """Calcula la pérdida y las métricas para train o validación."""

        aurora_batch = self.prepare_aurora_batch(mongo_batch)
        total_loss = 0
        n_steps = len(mongo_batch["targets"])

        for s in range(1, n_steps + 1):
            prediction = self(aurora_batch)
            target_key = f"step_{s}"

            t_u = mongo_batch["targets"][target_key]["10u"][..., self.lon_indices]
            t_v = mongo_batch["targets"][target_key]["10v"][..., self.lon_indices]
            t_2t = mongo_batch["targets"][target_key]["2t"][..., self.lon_indices]
            t_msl = mongo_batch["targets"][target_key]["msl"][..., self.lon_indices]

            mse_u = F.mse_loss(prediction.surf_vars["10u"], t_u)
            mse_v = F.mse_loss(prediction.surf_vars["10v"], t_v)
            mse_2t = F.mse_loss(prediction.surf_vars["2t"], t_2t)
            mse_msl = F.mse_loss(prediction.surf_vars["msl"] / 500.0, t_msl / 500.0)

            step_loss = (mse_u + mse_v) + 0.1 * (mse_2t + mse_msl)
            total_loss += step_loss

            rmse_step = torch.sqrt(mse_u + mse_v)
            step_mae = F.l1_loss(prediction.surf_vars["10u"], t_u) + F.l1_loss(
                prediction.surf_vars["10v"], t_v
            )

            self.log(f"{stage}/rmse_step_{s}", rmse_step, prog_bar=(s == 1))
            self.log(f"{stage}/mae_step_{s}", step_mae)

            if s < n_steps:
                new_surf_vars = {
                    k: torch.stack(
                        [
                            aurora_batch.surf_vars[k][:, 1],
                            prediction.surf_vars[k].squeeze(1),
                        ],
                        dim=1,
                    )
                    for k in ("2t", "10u", "10v", "msl")
                }
                aurora_batch.surf_vars = new_surf_vars

        self.log(f"{stage}/loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        """Ejecuta el paso de entrenamiento de Lightning."""

        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Ejecuta el paso de validación de Lightning."""

        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        """Define el optimizador y el scheduler del ajuste fino."""

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg_aurora["learning_rate"]),
            weight_decay=float(self.cfg_aurora["weight_decay"]),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg_aurora.get("epochs", 10),
            eta_min=float(self.cfg_aurora.get("scheduler_eta_min", 1e-6)),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
