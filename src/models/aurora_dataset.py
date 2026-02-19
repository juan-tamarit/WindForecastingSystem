import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from aurora.model.aurora import AuroraSmallPretrained
from aurora.batch import Batch, Metadata
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
import numpy as np

class AuroraMongoDataset(Dataset):
    def __init__(self, times, cfg_mdb, lats_torch, lons_torch, cfg_aurora):
        self.times = times  
        self.cfg_mdb = cfg_mdb
        self.cfg_aurora = cfg_aurora
        self.lats = lats_torch.tolist()
        self.lons = lons_torch.tolist()
        
        # IMPORTANTE: No abrimos conexión aquí para evitar errores en el multiprocessing
        self.client = None
        self.col = None
        
        self.lat_map = {round(float(lat), 2): i for i, lat in enumerate(self.lats)}
        self.lon_map = {round(float(lon), 2): j for j, lon in enumerate(self.lons)}
        
        # Cargamos orografía una sola vez (conexión local efímera)
        self.static_grid = self._load_static_grid()

    def _load_static_grid(self):
        """Carga datos constantes (elevación, lsm) y cierra la conexión al terminar."""
        client = MongoClient(self.cfg_mdb["uri"])
        col = client[self.cfg_mdb["db_name"]][self.cfg_mdb["collection_pro"]]
        
        n_lats, n_lons = len(self.lats), len(self.lons)
        elev = torch.zeros(n_lats, n_lons)
        lsm = torch.zeros(n_lats, n_lons)
        
        # Usamos el primer registro para extraer la máscara de tierra y elevación
        cursor = col.find({"valid_time": self.times[0]}, {"latitude":1, "longitude":1, "elevacion_m":1, "lsm":1})
        for doc in cursor:
            i, j = self.lat_map.get(doc['latitude']), self.lon_map.get(doc['longitude'])
            if i is not None and j is not None:
                elev[i, j] = doc.get('elevacion_m', 0)
                lsm[i, j] = doc.get('lsm', 0)
        
        client.close()
        return {"elevation": elev, "lsm": lsm}

    def __len__(self):
        h_hist = self.cfg_aurora["input_hours"]
        h_step = self.cfg_aurora["target_hours"]
        h_total = self.cfg_aurora["forecast_hours"]
        
        # Calculamos número de pasos (mínimo 1)
        n_steps = max(1, h_total // h_step)
        
        # El margen real es el historial + todos los pasos que vamos a predecir
        margin = h_hist + (h_step * n_steps)
        return len(self.times) - margin

    def _get_grid(self, time):
        """Recupera el grid usando la conexión propia del worker."""
        n_lats, n_lons = len(self.lats), len(self.lons)
        grids = {k: torch.zeros(n_lats, n_lons) for k in ["2t", "10u", "10v", "msl"]}
        
        cursor = self.col.find({"valid_time": time})
        for doc in cursor:
            i, j = self.lat_map.get(doc['latitude']), self.lon_map.get(doc['longitude'])
            if i is not None and j is not None:
                grids["2t"][i, j] = doc.get('t2m', 0)
                grids["10u"][i, j] = doc.get('u10', 0)
                grids["10v"][i, j] = doc.get('v10', 0)
                grids["msl"][i, j] = doc.get('msl', 0)
        return grids

    def __getitem__(self, idx):
        if self.client is None:
            self.client = MongoClient(self.cfg_mdb["uri"])
            self.col = self.client[self.cfg_mdb["db_name"]][self.cfg_mdb["collection_pro"]]
        
        h_hist = self.cfg_aurora["input_hours"]
        h_step = self.cfg_aurora["target_hours"]
        h_total = self.cfg_aurora["forecast_hours"]
        
        # Calculamos número de pasos (mínimo 1)
        n_steps = max(1, h_total // h_step)
        
        # Inputs (t_past y t_now)
        t_past = self.times[idx]
        t_now  = self.times[idx + h_hist]
        
        grid_past = self._get_grid(t_past)
        grid_now  = self._get_grid(t_now)
        
        inputs = {k: torch.stack([grid_past[k], grid_now[k]], dim=0) for k in grid_past.keys()}
        
        # Targets dinámicos
        targets = {}
        for s in range(1, n_steps + 1):
            t_target = self.times[idx + h_hist + (h_step * s)]
            grid_tgt = self._get_grid(t_target)
            targets[f"step_{s}"] = {k: grid_tgt[k].unsqueeze(0) for k in grid_tgt.keys()}
        
        return {"inputs": inputs, "targets": targets, "statics": self.static_grid}

class AuroraDataModule(pl.LightningDataModule):
    def __init__(self, cfg_mdb, cfg_aurora):
        super().__init__()
        self.cfg_mdb = cfg_mdb
        self.cfg_aurora = cfg_aurora
        self.batch_size = cfg_aurora["batch_size"]
        self.num_workers = cfg_aurora.get("num_workers", 4)
        self.lats = None
        self.lons = None

    def setup(self, stage=None):
        # 1. Conexión temporal para metadatos y cálculo de splits
        client = MongoClient(self.cfg_mdb["uri"])
        col = client[self.cfg_mdb["db_name"]][self.cfg_mdb["collection_pro"]]
    
        all_times = sorted(col.distinct("valid_time"))
        raw_lats = sorted(col.distinct("latitude"), reverse=True)
        raw_lons = sorted(col.distinct("longitude"))
        
        # Recorte de seguridad para la arquitectura de Aurora (32x56)
        self.lats = torch.tensor(raw_lats[:32], dtype=torch.float32)
        self.lons = torch.tensor(raw_lons[:56], dtype=torch.float32)
    
        # 2. CERRAMOS LA CONEXIÓN (El proceso principal no debe mantenerla)
        client.close()
    
        # 3. SPLIT DINÁMICO (Desde YAML) CON PROTECCIÓN DE TEST
        n_total = len(all_times)
        
        # Reservamos el 15% final para Test (intocable durante el entrenamiento)
        test_size = int(n_total * 0.15)
        remaining_size = n_total - test_size
        
        # El resto se divide según el train_split del YAML
        split_val = self.cfg_aurora["train_split"]
        train_end = int(remaining_size * split_val)

        self.train_times = all_times[:train_end]
        self.val_times = all_times[train_end:remaining_size]
        self.test_times = all_times[remaining_size:]
        
        print(f"--- DataModule Configurado ---")
        print(f"Split Train/Val (YAML): {split_val}")
        print(f"Fechas Train: {self.train_times[0].date()} a {self.train_times[-1].date()}")
        print(f"Fechas Val:   {self.val_times[0].date()} a {self.val_times[-1].date()}")
        print(f"Fechas Test:  {self.test_times[0].date()} a {self.test_times[-1].date()}")

    def train_dataloader(self):
        return DataLoader(
            AuroraMongoDataset(self.train_times, self.cfg_mdb, self.lats, self.lons, self.cfg_aurora),
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            AuroraMongoDataset(self.val_times, self.cfg_mdb, self.lats, self.lons, self.cfg_aurora),
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            AuroraMongoDataset(self.test_times, self.cfg_mdb, self.lats, self.lons, self.cfg_aurora),
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

class AuroraFinetuner(pl.LightningModule):
    def __init__(self, cfg_coords, cfg_aurora):
        super().__init__()
        self.save_hyperparameters(cfg_aurora)
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
        device = mongo_dict["inputs"]["2t"].device
        current_indices = self.lon_indices.to(device)

        surf_vars = {
            k: mongo_dict["inputs"][k][..., current_indices] # <--- Usamos los índices en el sitio correcto
            for k in ("2t", "10u", "10v", "msl")
        }

        # Repetimos lo mismo para las estáticas
        z_reordered = mongo_dict["statics"]["elevation"][..., current_indices]
        lsm_reordered = mongo_dict["statics"]["lsm"][..., current_indices]

        static_vars = {
            "z": z_reordered.mean(dim=0),
            "lsm": lsm_reordered.mean(dim=0)
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
                atmos_levels=(100, 250, 500, 850)
            )
        )

    def forward(self, batch):
        return self.model(batch)

    def shared_step(self, mongo_batch, stage):
        # 1. Preparar primer batch y predecir
        aurora_batch = self.prepare_aurora_batch(mongo_batch)
        
        total_loss = 0
        n_steps = len(mongo_batch["targets"])
        
        for s in range(1, n_steps + 1):
            # Predicción del paso actual
            prediction = self(aurora_batch)
            
            # Ground Truth del paso actual
            target_key = f"step_{s}"
            t_u = mongo_batch["targets"][target_key]["10u"][..., self.lon_indices]
            t_v = mongo_batch["targets"][target_key]["10v"][..., self.lon_indices]
            
            # Cálculo de pérdida (acumulada)
            step_loss = F.mse_loss(prediction.surf_vars["10u"], t_u) + \
                        F.mse_loss(prediction.surf_vars["10v"], t_v)
            total_loss += step_loss
            
            # Log individual por paso
            self.log(f"{stage}/rmse_step_{s}", torch.sqrt(step_loss), prog_bar=(s==1))

            # Preparar siguiente paso autoregresivo (si no es el último)
            if s < n_steps:
                new_surf_vars = {
                    k: torch.stack([
                        aurora_batch.surf_vars[k][:, 1], # El 'now' anterior pasa a ser 'past'
                        prediction.surf_vars[k].squeeze(1) # La predicción actual pasa a ser 'now'
                    ], dim=1)
                    for k in ("2t", "10u", "10v", "msl")
                }
                aurora_batch.surf_vars = new_surf_vars
        
        # Log de pérdida total
        self.log(f"{stage}/loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.cfg_aurora["learning_rate"]),
            weight_decay=float(self.cfg_aurora["weight_decay"])
        )
    
        # scheduler_t_max debería ser igual al número de épocas totales
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg_aurora.get("epochs", 10), 
            eta_min=float(self.cfg_aurora.get("scheduler_eta_min", 1e-6))
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Se actualiza cada vez que termina una época
            },
        }