import torch
import glob
import os
import pandas as pd
import torch.nn.functional as F
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.config import MDB, PARAMS

def extract_loss(path):
    try:
        
        name = os.path.basename(path).replace(".ckpt", "")
        last_part = name.split("-")[-1] 
        
        
        if "=" in last_part:
            value = last_part.split("=")[-1]
        else:
            value = last_part
            
        return float(value)
    except:
        return float('inf')

def get_best_checkpoint(checkpoint_dir="checkpoints/"):
    
    ckpts = glob.glob(os.path.join(checkpoint_dir, "aurora-*.ckpt"))
    
    ckpts = [c for c in ckpts if "last" not in c]
    
    if not ckpts:
        return None

    
    return min(ckpts, key=extract_loss)

def run_test():
    FASE_OBJETIVO = "fase2"

    cfg_test = PARAMS["aurora_base"].copy()
    conf_fase = PARAMS["fases"][FASE_OBJETIVO]

    cfg_test.update({
        "epochs": conf_fase["epochs"],
        "target_hours": conf_fase["target_hours"],
        "forecast_hours": conf_fase["forecast_hours"],
        "learning_rate": conf_fase["learning_rates"][-1]
    })

    dm = AuroraDataModule(cfg_mdb=MDB, cfg_aurora=cfg_test)
    dm.setup()
    test_loader = dm.test_dataloader()

    checkpoint_path = get_best_checkpoint()
    if not checkpoint_path:
        print(f"No se han encontrado checkpoints. Revisa la ruta del directorio.")
        return

    print(f"Mejor modelo detectado para Test: {os.path.basename(checkpoint_path)}")
    
    model = AuroraFinetuner.load_from_checkpoint(
        checkpoint_path,
        cfg_coords={"lats": dm.lats, "lons": dm.lons},
        cfg_aurora=cfg_test
    )
    model.eval().cuda()

    results = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            def to_cuda(obj):
                if isinstance(obj, dict): return {k: to_cuda(v) for k, v in obj.items()}
                return obj.cuda() if torch.is_tensor(obj) else obj
            
            batch = to_cuda(batch)

            # --- PREPARACIÓN DIRECTA ---
            # Aurora ya recibe los datos tal cual vienen de MongoDB (K, m/s, Pa)
            aurora_batch = model.prepare_aurora_batch(batch)
            
            # Persistencia: Usamos el último valor del input (t=1) en unidades reales
            # Reordenado por índices de longitud igual que la predicción
            per_u_real = batch["inputs"]["10u"][:, 1][..., model.lon_indices]
            per_v_real = batch["inputs"]["10v"][:, 1][..., model.lon_indices]

            n_steps = len(batch["targets"])
            for s in range(1, n_steps + 1):
                prediction = model(aurora_batch)
                target_key = f"step_{s}"
                
                # Targets y Predicciones en UNIDADES REALES
                t_u_real = batch["targets"][target_key]["10u"][..., model.lon_indices].squeeze(1)
                t_v_real = batch["targets"][target_key]["10v"][..., model.lon_indices].squeeze(1)
                
                p_u_real = prediction.surf_vars["10u"].squeeze(1)
                p_v_real = prediction.surf_vars["10v"].squeeze(1)

                # --- MÉTRICAS ---
                # RMSE Aurora
                rmse_aur = torch.sqrt(F.mse_loss(p_u_real, t_u_real) + F.mse_loss(p_v_real, t_v_real))
                # MAE Aurora
                mae_aur = F.l1_loss(p_u_real, t_u_real) + F.l1_loss(p_v_real, t_v_real)

                # RMSE Persistencia
                rmse_per = torch.sqrt(F.mse_loss(per_u_real, t_u_real) + F.mse_loss(per_v_real, t_v_real))

                results.append({
                    "batch_idx": i,
                    "step": s,
                    "rmse_aurora": rmse_aur.item(),
                    "mae_aurora": mae_aur.item(),
                    "rmse_persist": rmse_per.item()
                })

                if s < n_steps:
                    # Autoregresión pasando predicciones físicas al siguiente paso
                    new_surf_vars = {
                        k: torch.stack([
                            aurora_batch.surf_vars[k][:, 1], 
                            prediction.surf_vars[k].squeeze(1) 
                        ], dim=1)
                        for k in ("2t", "10u", "10v", "msl")
                    }
                    aurora_batch.surf_vars = new_surf_vars

    # --- PROCESAMIENTO DE RESULTADOS ---
    df = pd.DataFrame(results)
    summary = df.groupby('step').agg({
        'rmse_aurora': 'mean',
        'mae_aurora': 'mean',
        'mape_aurora': 'mean',
        'rmse_persist': 'mean'
    })
    
    # Skill Score: ¿Cuánto mejor somos que la persistencia? (%)
    summary['skill_score'] = ((summary['rmse_persist'] - summary['rmse_aurora']) / summary['rmse_persist']) * 100

    output_dir = "docs/resultados"
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(checkpoint_path).replace(".ckpt", "")
    csv_path = os.path.join(output_dir, f"test_results_{model_name}.csv")

    print("\n" + "="*60)
    print(f"MÉTRICAS FINALES DE TEST (UNIDADES FÍSICAS)")
    print("-" * 60)
    print(summary.to_string())
    print("="*60)
    
    df.to_csv(csv_path, index=False)
    print(f"CSV de test guardado en: {csv_path}")

if __name__ == "__main__":
    run_test()