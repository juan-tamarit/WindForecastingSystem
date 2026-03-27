"""Script de evaluación final del modelo Aurora.

Este punto de entrada localiza el mejor checkpoint disponible, ejecuta el test
autoregresivo sobre el conjunto temporal reservado y guarda las métricas
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
    """Orquesta la evaluación final del modelo sobre el conjunto de test."""

    def extract_loss(self, path):
        """Extrae la pérdida codificada en el nombre de un checkpoint."""

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
        """Devuelve el checkpoint con mejor pérdida dentro del directorio dado."""

        ckpts = glob.glob(os.path.join(checkpoint_dir, "aurora-*.ckpt"))
        ckpts = [checkpoint for checkpoint in ckpts if "last" not in checkpoint]

        if not ckpts:
            return None

        return min(ckpts, key=self.extract_loss)

    def run(self):
        """Ejecuta la evaluación completa y guarda los resultados en CSV."""

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

        model = AuroraFinetuner.load_from_checkpoint(
            checkpoint_path,
            cfg_coords={"lats": dm.lats, "lons": dm.lons},
            cfg_aurora=cfg_test,
        )
        model.eval().cuda()

        results = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                def to_cuda(obj):
                    """Mueve recursivamente tensores del batch a GPU."""

                    if isinstance(obj, dict):
                        return {k: to_cuda(v) for k, v in obj.items()}
                    return obj.cuda() if torch.is_tensor(obj) else obj

                batch = to_cuda(batch)
                aurora_batch = model.prepare_aurora_batch(batch)

                per_u_real = batch["inputs"]["10u"][:, 1][..., model.lon_indices]
                per_v_real = batch["inputs"]["10v"][:, 1][..., model.lon_indices]

                n_steps = len(batch["targets"])
                for s in range(1, n_steps + 1):
                    prediction = model(aurora_batch)
                    target_key = f"step_{s}"

                    t_u_real = batch["targets"][target_key]["10u"][..., model.lon_indices].squeeze(1)
                    t_v_real = batch["targets"][target_key]["10v"][..., model.lon_indices].squeeze(1)

                    p_u_real = prediction.surf_vars["10u"].squeeze(1)
                    p_v_real = prediction.surf_vars["10v"].squeeze(1)

                    rmse_aur = torch.sqrt(F.mse_loss(p_u_real, t_u_real) + F.mse_loss(p_v_real, t_v_real))
                    mae_aur = F.l1_loss(p_u_real, t_u_real) + F.l1_loss(p_v_real, t_v_real)
                    rmse_per = torch.sqrt(F.mse_loss(per_u_real, t_u_real) + F.mse_loss(per_v_real, t_v_real))

                    results.append(
                        {
                            "batch_idx": i,
                            "step": s,
                            "rmse_aurora": rmse_aur.item(),
                            "mae_aurora": mae_aur.item(),
                            "rmse_persist": rmse_per.item(),
                        }
                    )

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

        df = pd.DataFrame(results)
        summary = df.groupby("step").agg(
            {
                "rmse_aurora": "mean",
                "mae_aurora": "mean",
                "rmse_persist": "mean",
            }
        )

        summary["skill_score"] = (
            (summary["rmse_persist"] - summary["rmse_aurora"])
            / summary["rmse_persist"]
        ) * 100

        output_dir = "docs/resultados"
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.basename(checkpoint_path).replace(".ckpt", "")
        csv_path = os.path.join(output_dir, f"test_results_{model_name}.csv")

        print("\n" + "=" * 60)
        print("MÃ‰TRICAS FINALES DE TEST (UNIDADES FÃSICAS)")
        print("-" * 60)
        print(summary.to_string())
        print("=" * 60)

        df.to_csv(csv_path, index=False)
        print(f"CSV de test guardado en: {csv_path}")


def extract_loss(path):
    """Mantiene la API histórica basada en función para la pérdida del checkpoint."""

    return AuroraTester().extract_loss(path)


def get_best_checkpoint(checkpoint_dir="checkpoints/"):
    """Mantiene la API histórica basada en función para localizar checkpoints."""

    return AuroraTester().get_best_checkpoint(checkpoint_dir=checkpoint_dir)


def run_test():
    """Mantiene la API histórica basada en función para lanzar el test."""

    return AuroraTester().run()


if __name__ == "__main__":
    run_test()
