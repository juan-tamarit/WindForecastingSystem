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
    dm = AuroraDataModule(cfg_mdb=MDB, cfg_aurora=PARAMS["aurora"])
    dm.setup()
    test_loader = dm.test_dataloader()

    checkpoint_path = get_best_checkpoint()
    if not checkpoint_path:
        print(f"No se han encontrado checkpoints en /checkpoints.")
        return

    print(f"Mejor modelo detectado: {os.path.basename(checkpoint_path)}")
    
    model = AuroraFinetuner.load_from_checkpoint(
        checkpoint_path,
        cfg_coords={"lats": dm.lats, "lons": dm.lons},
        cfg_aurora=PARAMS["aurora"]
    )
    model.eval().cuda()

    results = []
    eps = 1e-5
    print("Iniciando inferencia autorregresiva (24h) sobre conjunto de Test...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            def to_cuda(obj):
                if isinstance(obj, dict): return {k: to_cuda(v) for k, v in obj.items()}
                return obj.cuda() if torch.is_tensor(obj) else obj
            
            batch = to_cuda(batch)

            aurora_batch = model.prepare_aurora_batch(batch)
            
            
            persist_u = batch["inputs"]["10u"][:, 1][..., model.lon_indices]
            persist_v = batch["inputs"]["10v"][:, 1][..., model.lon_indices]

            
            n_steps = len(batch["targets"])
            for s in range(1, n_steps + 1):
                
                prediction = model(aurora_batch)
                
                target_key = f"step_{s}"
                t_u = batch["targets"][target_key]["10u"][..., model.lon_indices].squeeze(1)
                t_v = batch["targets"][target_key]["10v"][..., model.lon_indices].squeeze(1)
                
                pred_u = prediction.surf_vars["10u"].squeeze(1)
                pred_v = prediction.surf_vars["10v"].squeeze(1)

                
                mse_step = F.mse_loss(pred_u, t_u) + F.mse_loss(pred_v, t_v)
                rmse_aur = torch.sqrt(mse_step)
                
                
                mae_aur = F.l1_loss(pred_u, t_u) + F.l1_loss(pred_v, t_v)
                
                
                mape_u = torch.mean(torch.abs((t_u - pred_u) / (t_u + eps)))
                mape_v = torch.mean(torch.abs((t_v - pred_v) / (t_v + eps)))
                mape_aur = (mape_u + mape_v) / 2 * 100

                rmse_per = torch.sqrt(F.mse_loss(persist_u, t_u) + F.mse_loss(persist_v, t_v))

                results.append({
                    "batch_idx": i,
                    "step": s,
                    "rmse_aurora": rmse_aur.item(),
                    "mae_aurora": mae_aur.item(),
                    "mape_aurora": mape_aur.item(),
                    "rmse_persist": rmse_per.item()
                })

                if s < n_steps:
                    new_surf_vars = {
                        k: torch.stack([
                            aurora_batch.surf_vars[k][:, 1],   
                            prediction.surf_vars[k].squeeze(1) 
                        ], dim=1)
                        for k in ("2t", "10u", "10v", "msl")
                    }
                    aurora_batch.surf_vars = new_surf_vars

    df = pd.DataFrame(results)
    
    summary = df.groupby('step').agg({
        'rmse_aurora': 'mean',
        'mae_aurora': 'mean',
        'mape_aurora': 'mean',
        'rmse_persist': 'mean'
    })
    
    summary['skill_score'] = ((summary['rmse_persist'] - summary['rmse_aurora']) / summary['rmse_persist']) * 100

    output_dir = "docs/resultados"
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(checkpoint_path).replace(".ckpt", "")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(output_dir, f"test_full_24h_{model_name}_{timestamp}.csv")

    print("\n" + "="*60)
    print(f"MÉTRICAS FINALES DE TEST (HORIZONTE 24H)")
    print("-" * 60)
    print(summary.to_string())
    print("="*60)
    
    df.to_csv(csv_path, index=False)
    print(f"Dataset completo de test guardado en: {csv_path}")

if __name__ == "__main__":
    run_test()