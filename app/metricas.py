import math
import torch
import matplotlib.pyplot as plt

"""
Calcula MAE, RMSE y MAPE para un target concreto (por índice) sobre el
dataset de validación.

Parámetros
----------
model : TemporalFusionTransformer
    Modelo ya entrenado y cargado (best_tft).
validation : TimeSeriesDataSet
    Dataset de validación (no DataLoader).
target_idx : int
    Índice del target en la lista 'targets' usada al crear el modelo.
    Ejemplo: 0 -> wind_speed, 1 -> wind_dir_sin, 2 -> wind_dir_cos.
batch_size : int
    Tamaño de batch para el DataLoader de validación.

Devuelve
--------
dict
    Diccionario con claves 'MAE', 'RMSE', 'MAPE'.
"""

def evaluateTarget(model, validation, target_idx, batch_size):
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0
    )

    metrics_pred = model.predict(
        val_dataloader,
        return_y=True,
        trainer_kwargs=dict(accelerator="cpu")
    )

    # Salida de la red y conversión a predicción puntual
    y_pred_net = metrics_pred.output
    loss_metric = model.loss
    y_pred_list = loss_metric.to_prediction(y_pred_net)  # lista de tensores por target

    # predicciones del target elegido
    y_pred_t = y_pred_list[target_idx].float().reshape(-1)

    # valores reales: primer elemento del tuple, índice del target
    y_true_raw = metrics_pred.y
    # y_true_raw[0] es un tuple de 3 tensores (uno por target)
    y_true_tensor = y_true_raw[0][target_idx]
    y_true_t = y_true_tensor.float().reshape(-1)

    # asegurar misma longitud
    n = min(len(y_pred_t), len(y_true_t))
    y_pred_t = y_pred_t[:n]
    y_true_t = y_true_t[:n]

    mae = torch.mean(torch.abs(y_true_t - y_pred_t)).item()
    rmse = math.sqrt(torch.mean((y_true_t - y_pred_t) ** 2).item())
    epsilon = 1e-6
    mape = torch.mean(
        torch.abs((y_true_t - y_pred_t) / (y_true_t + epsilon))
    ).item() * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

"""
Dibuja n_examples ejemplos de serie real vs predicha
usando plot_prediction de PyTorch Forecasting.
"""

def plotPredictions(model, validation, batch_size, n_examples):
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
    )

    predictions = model.predict(
        val_dataloader,
        return_x=True,
        mode="raw",
        trainer_kwargs=dict(accelerator="cpu"),
    )

    x_raw = predictions.x

    for idx in range(n_examples):
        figs = model.plot_prediction(
            x_raw,
            predictions.output,
            idx=idx,
            add_loss_to_title=True,
            show_future_observed=True,
        )
        for f in figs:
            f.show()
"""
Dibuja el histograma de errores (y_true - y_pred) para un target concreto.
target_idx sigue el mismo orden que la lista 'targets' usada al entrenar.
"""
def plotErrorHistogram(model, validation, batch_size, target_idx):
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
    )

    metrics_pred = model.predict(
        val_dataloader,
        return_y=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    # predicciones puntuales por target
    y_pred_net = metrics_pred.output
    loss_metric = model.loss
    y_pred_list = loss_metric.to_prediction(y_pred_net)

    y_pred_t = y_pred_list[target_idx].float().reshape(-1)

    # verdad a partir de metrics_pred.y (misma estructura que en evaluate_target)
    y_true_raw = metrics_pred.y
    y_true_tensor = y_true_raw[0][target_idx]
    y_true_t = y_true_tensor.float().reshape(-1)

    n = min(len(y_pred_t), len(y_true_t))
    errors = (y_true_t[:n] - y_pred_t[:n]).detach().cpu().numpy()

    plt.hist(errors, bins=50)
    plt.xlabel("Error (y_true - y_pred)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de errores en validación")
    plt.tight_layout()
    plt.show()