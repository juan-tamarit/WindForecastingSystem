import torch
import glob
import os
import pandas as pd
import torch.nn.functional as F
from src.models.aurora_dataset import AuroraDataModule, AuroraFinetuner
from src.config import MDB, PARAMS

def extract_loss(path):
    """Extrae el valor numérico del loss sin importar el formato del nombre."""
    try:
        # Quitamos extensión y tomamos la última parte tras el último guion
        name = os.path.basename(path).replace(".ckpt", "")
        last_part = name.split("-")[-1] 
        
        # Si contiene un '=', nos quedamos con lo de la derecha (ej: val_loss=0.001)
        if "=" in last_part:
            value = last_part.split("=")[-1]
        else:
            value = last_part
            
        return float(value)
    except:
        return float('inf')

def get_best_checkpoint(checkpoint_dir="checkpoints/"):
    # Buscamos ambos patrones para que no te falle ahora
    ckpts = glob.glob(os.path.join(checkpoint_dir, "aurora-*.ckpt"))
    # Filtramos para ignorar el 'last.ckpt'
    ckpts = [c for c in ckpts if "last" not in c]
    
    if not ckpts:
        return None

    # Devolvemos el que tenga el loss más bajo (ignorando los 0.0000 si hay mejores)
    # Si todos son 0.0000, simplemente devolverá el primero que encuentre
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
    print("Iniciando inferencia sobre conjunto de Test...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Mover batch a GPU recursivamente
            def to_cuda(obj):
                if isinstance(obj, dict): return {k: to_cuda(v) for k, v in obj.items()}
                return obj.cuda() if torch.is_tensor(obj) else obj
            
            batch = to_cuda(batch)

            # Predicción Aurora (Paso 1)
            aurora_batch_1 = model.prepare_aurora_batch(batch)
            pred_1 = model(aurora_batch_1)
            
            # Ground Truth (Reordenado con lon_indices)
            t1_u = batch["targets"]["step_1"]["10u"][..., model.lon_indices].squeeze(1)
            t1_v = batch["targets"]["step_1"]["10v"][..., model.lon_indices].squeeze(1)
            
            # Persistencia (Valor en 't_now')
            persist_u = batch["inputs"]["10u"][:, 1][..., model.lon_indices]
            persist_v = batch["inputs"]["10v"][:, 1][..., model.lon_indices]

            # RMSE combinado (U + V)
            rmse_aur = torch.sqrt(F.mse_loss(pred_1.surf_vars["10u"].squeeze(1), t1_u) + 
                                  F.mse_loss(pred_1.surf_vars["10v"].squeeze(1), t1_v))
            
            rmse_per = torch.sqrt(F.mse_loss(persist_u, t1_u) + 
                                  F.mse_loss(persist_v, t1_v))

            results.append({"rmse_aurora": rmse_aur.item(), "rmse_persist": rmse_per.item()})

    # Informe de resultados
    df = pd.DataFrame(results)
    avg_aur = df['rmse_aurora'].mean()
    avg_per = df['rmse_persist'].mean()
    skill = ((avg_per - avg_aur) / avg_per) * 100

    # Crear carpeta de resultados si no existe
    output_dir = "docs/resultados"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre de archivo con fecha y nombre del modelo para no sobrescribir
    model_name = os.path.basename(checkpoint_path).replace(".ckpt", "")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(output_dir, f"test_{model_name}_{timestamp}.csv")

    print("\n" + "="*45)
    print(f"MÉTRICAS DE VALIDACIÓN FINAL (TEST)")
    print("-" * 45)
    print(f"Modelo evaluado:    {model_name}")
    print(f"Distancia RMSE Aur: {avg_aur:.6f}")
    print(f"Distancia RMSE Per: {avg_per:.6f}")
    print(f"Skill Score:        {skill:.2f}%")
    print("="*45)
    
    df.to_csv(csv_path, index=False)
    print(f"Resultados guardados en: {csv_path}")

if __name__ == "__main__":
    run_test()