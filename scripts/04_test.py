"""Script de evaluacion final del modelo Aurora.

Este punto de entrada localiza el mejor checkpoint disponible, ejecuta el test
autoregresivo sobre el conjunto temporal reservado y guarda las metricas
agregadas y detalladas del experimento.
"""

import glob
import os

import pandas as pd
import torch
import torch.nn.functional as F

from src.config import MDB, PARAMS
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner


class AuroraTester:
    """Orquesta la evaluacion final del modelo sobre el conjunto de test."""

    def extract_loss(self, path):
        """Extrae la perdida codificada en el nombre de un checkpoint."""

        try:
            name = os.path.basename(path).replace(".ckpt", "")
            last_part = name.split("-")[-1]

            if "=" in last_part:
                value = last_part.split("=")[-1]
            else:
                value = last_part

            return float(value)
        except Exception:
            return float("inf")

    def get_best_checkpoint(self, checkpoint_dir="checkpoints/"):
        """Devuelve el checkpoint con mejor perdida dentro del directorio dado."""

        ckpts = glob.glob(os.path.join(checkpoint_dir, "aurora-*.ckpt"))
        ckpts = [checkpoint for checkpoint in ckpts if "last" not in checkpoint]

        if not ckpts:
            return None

        return min(ckpts, key=self.extract_loss)

    def to_cuda(self, obj):
        """Mueve recursivamente tensores del batch a GPU."""

        if isinstance(obj, dict):
            return {k: self.to_cuda(v) for k, v in obj.items()}
        return obj.cuda() if torch.is_tensor(obj) else obj

    def update_autoregressive_batch(self, aurora_batch, prediction):
        """Actualiza el estado autoregresivo usando la prediccion del paso actual."""

        aurora_batch.surf_vars = {
            k: torch.stack(
                [
                    aurora_batch.surf_vars[k][:, 1],
                    prediction.surf_vars[k].squeeze(1),
                ],
                dim=1,
            )
            for k in ("2t", "10u", "10v", "msl")
        }

    def compute_wind_rmse(self, pred_u, pred_v, target_u, target_v):
        """Calcula el RMSE combinado de 10u y 10v."""

        return torch.sqrt(F.mse_loss(pred_u, target_u) + F.mse_loss(pred_v, target_v))

    def run(self):
        """Ejecuta la evaluacion completa y guarda los resultados en CSV."""

        fase_objetivo = "fase3"

        cfg_test = PARAMS["aurora_base"].copy()
        conf_fase = PARAMS["fases"][fase_objetivo]

        cfg_test.update(
            {
                "epochs": conf_fase["epochs"],
                "target_hours": conf_fase["target_hours"],
                "forecast_hours": conf_fase["forecast_hours"],
                "learning_rate": conf_fase["learning_rates"][-1],
            }
        )

        dm = AuroraDataModule(cfg_mdb=MDB, cfg_aurora=cfg_test)
        dm.setup()
        test_loader = dm.test_dataloader()

        checkpoint_path = self.get_best_checkpoint()
        if not checkpoint_path:
            print("No se han encontrado checkpoints. Revisa la ruta del directorio.")
            return

        print(f"Mejor modelo detectado para Test: {os.path.basename(checkpoint_path)}")

        model_ft = AuroraFinetuner.load_from_checkpoint(
            checkpoint_path,
            cfg_coords={"lats": dm.lats, "lons": dm.lons},
            cfg_aurora=cfg_test,
        )
        model_base = AuroraFinetuner(
            cfg_coords={"lats": dm.lats, "lons": dm.lons},
            cfg_aurora=cfg_test,
        )

        model_ft.eval().cuda()
        model_base.eval().cuda()
        torch.cuda.empty_cache()

        clima_stats = dm.train_dataset.stats
        clima_u = torch.as_tensor(
            clima_stats["10u"]["mean"], dtype=torch.float32, device="cuda"
        )[..., model_ft.lon_indices]
        clima_v = torch.as_tensor(
            clima_stats["10v"]["mean"], dtype=torch.float32, device="cuda"
        )[..., model_ft.lon_indices]

        results = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = self.to_cuda(batch)

                batch_ft = model_ft.prepare_aurora_batch(batch)
                batch_base = model_base.prepare_aurora_batch(batch)

                per_u_real = batch["inputs"]["10u"][:, 1][..., model_ft.lon_indices]
                per_v_real = batch["inputs"]["10v"][:, 1][..., model_ft.lon_indices]

                n_steps = len(batch["targets"])
                for s in range(1, n_steps + 1):
                    target_key = f"step_{s}"

                    prediction_ft = model_ft(batch_ft)
                    prediction_base = model_base(batch_base)

                    t_u_real = batch["targets"][target_key]["10u"][..., model_ft.lon_indices].squeeze(1)
                    t_v_real = batch["targets"][target_key]["10v"][..., model_ft.lon_indices].squeeze(1)

                    p_u_ft = prediction_ft.surf_vars["10u"].squeeze(1)
                    p_v_ft = prediction_ft.surf_vars["10v"].squeeze(1)
                    p_u_base = prediction_base.surf_vars["10u"].squeeze(1)
                    p_v_base = prediction_base.surf_vars["10v"].squeeze(1)

                    clima_u_real = clima_u.unsqueeze(0).expand_as(t_u_real)
                    clima_v_real = clima_v.unsqueeze(0).expand_as(t_v_real)

                    rmse_ft = self.compute_wind_rmse(p_u_ft, p_v_ft, t_u_real, t_v_real)
                    rmse_base = self.compute_wind_rmse(p_u_base, p_v_base, t_u_real, t_v_real)
                    rmse_persist = self.compute_wind_rmse(per_u_real, per_v_real, t_u_real, t_v_real)
                    rmse_clima = self.compute_wind_rmse(clima_u_real, clima_v_real, t_u_real, t_v_real)
                    mae_ft = F.l1_loss(p_u_ft, t_u_real) + F.l1_loss(p_v_ft, t_v_real)

                    results.append(
                        {
                            "batch_idx": i,
                            "step": s,
                            "rmse_aurora_ft": rmse_ft.item(),
                            "rmse_aurora_base": rmse_base.item(),
                            "rmse_persist": rmse_persist.item(),
                            "rmse_clima": rmse_clima.item(),
                            "mae_aurora_ft": mae_ft.item(),
                        }
                    )

                    if s < n_steps:
                        self.update_autoregressive_batch(batch_ft, prediction_ft)
                        self.update_autoregressive_batch(batch_base, prediction_base)

        df = pd.DataFrame(results)
        summary = df.groupby("step").agg(
            {
                "rmse_aurora_ft": "mean",
                "rmse_aurora_base": "mean",
                "rmse_persist": "mean",
                "rmse_clima": "mean",
                "mae_aurora_ft": "mean",
            }
        )

        summary["skill_vs_persist"] = (
            (summary["rmse_persist"] - summary["rmse_aurora_ft"]) / summary["rmse_persist"]
        ) * 100
        summary["skill_vs_clima"] = (
            (summary["rmse_clima"] - summary["rmse_aurora_ft"]) / summary["rmse_clima"]
        ) * 100
        summary["skill_vs_base"] = (
            (summary["rmse_aurora_base"] - summary["rmse_aurora_ft"]) / summary["rmse_aurora_base"]
        ) * 100

        output_dir = "docs/resultados"
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.basename(checkpoint_path).replace(".ckpt", "")
        csv_path = os.path.join(output_dir, f"test_results_{model_name}.csv")

        print("\n" + "=" * 60)
        print("METRICAS FINALES DE TEST (UNIDADES FISICAS)")
        print("-" * 60)
        print(summary.to_string())
        print("=" * 60)

        df.to_csv(csv_path, index=False)
        print(f"CSV de test guardado en: {csv_path}")


def extract_loss(path):
    """Mantiene la API historica basada en funcion para la perdida del checkpoint."""

    return AuroraTester().extract_loss(path)


def get_best_checkpoint(checkpoint_dir="checkpoints/"):
    """Mantiene la API historica basada en funcion para localizar checkpoints."""

    return AuroraTester().get_best_checkpoint(checkpoint_dir=checkpoint_dir)


def run_test():
    """Mantiene la API historica basada en funcion para lanzar el test."""

    return AuroraTester().run()


if __name__ == "__main__":
    run_test()
