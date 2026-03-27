# Wind Forecasting with ERA5 and Aurora

This repository contains a Bachelor's Thesis project focused on wind
forecasting from ERA5 meteorological data. The project implements a complete
pipeline for data ingestion, storage, preprocessing, Aurora fine-tuning, and
final evaluation against a persistence baseline.

## Objective

The main goal of the project is to build a reproducible system to:

- download meteorological variables from ERA5 over the Iberian Peninsula;
- store them locally in MongoDB;
- transform them into a format suitable for spatio-temporal learning;
- fine-tune a pretrained Aurora model;
- evaluate predictive skill at increasing forecast horizons.

## Project Workflow

The final Bachelor's Thesis pipeline is executed from the `scripts` folder and
follows this order:

1. `scripts/01_load_data.py`
   Loads static fields and ERA5 time series into MongoDB.
2. `scripts/02_process_data.py`
   Builds the processed dataset collection with derived variables.
3. `scripts/03_finetune_aurora.py`
   Runs Aurora fine-tuning across phases and learning rates.
4. `scripts/04_test.py`
   Evaluates the best available checkpoint and generates test metrics.

The project also includes supporting utilities:

- `src/plots/plot_result.py` to generate a summary figure from the test CSV.
- `src/models/visualizer.py` to save metrics and images during training.

## Main Structure

```text
.
|-- checkpoints/            # Training checkpoints by phase and learning rate
|-- configs/
|   `-- config.yaml         # Main project configuration
|-- data/                   # Local MongoDB directory
|-- docs/
|   |-- entrenamiento/      # Metrics and images saved per epoch
|   `-- resultados/         # Final test results and plots
|-- scripts/                # Pipeline entry points
`-- src/
    |-- config.py
    |-- data_loading/
    |-- db/
    |-- frame/
    |-- models/
    `-- plots/
```

## Code Components

### Configuration

- `src/config.py`
  Loads environment variables, creates the CDS client, and exposes the main
  configuration read from `configs/config.yaml`.

### Data Loading

- `src/data_loading/era5.py`
  Downloads ERA5 time series and static fields, then transforms the downloaded
  files into structures ready to be stored in MongoDB.

- `src/utils.py`
  Orchestrates ingestion over temporal blocks and spatial grid points.

### Persistence and Processing

- `src/db/DBmanager.py`
  Manages MongoDB access and inserts ERA5 observations and static fields.

- `src/frame/DFmanager.py`
  Retrieves data from MongoDB and builds the processed dataset with derived
  variables such as `elevacion_m`, `time_idx`, and `location_id`.

### Modeling

- `src/models/aurora_dataset.py`
  Defines the `Dataset`, `LightningDataModule`, and `LightningModule` used for
  Aurora fine-tuning.

- `src/models/visualizer.py`
  Stores per-epoch metrics and a simple visual comparison between target and
  prediction.

### Final Visualization

- `src/plots/plot_result.py`
  Generates a final infographic with RMSE, MAE, and skill score against the
  persistence baseline.

## Experimental Configuration

The active configuration is stored in `configs/config.yaml`.

### Temporal Domain

- Start: `2024-01-01`
- End: `2025-12-31`

### Spatial Domain

- Latitudes: from `36.0` to `44.0`
- Longitudes: from `-10.0` to `4.0`
- Resolution: `0.25º`

### Training by Phases

Fine-tuning is organized into three phases:

- `fase1`
  Initial adaptation of the model to the task.
- `fase2`
  Stabilization and short autoregressive forecasting.
- `fase3`
  Final refinement for longer forecast horizons.

Each phase defines:

- maximum number of epochs;
- list of learning rates;
- target horizon (`target_hours`);
- total forecast horizon (`forecast_hours`);
- `min_delta` for early stopping.

## Requirements

At minimum, the project requires:

- Python
- a local MongoDB instance
- access to the Copernicus Climate Data Store API
- the Python dependencies used by the project

The main libraries used include:

- `pandas`
- `numpy`
- `pymongo`
- `xarray`
- `torch`
- `pytorch-lightning`
- `cdsapi`

## Environment Variables

The project expects a `.env` file with the required credentials. At least:

```env
AEMET_API_KEY=...
CDS_API_KEY=...
```

The final Bachelor's Thesis pipeline relies on ERA5 and MongoDB. AEMET access
is not part of the main workflow used in the final experiment.

## Execution

### 1. ERA5 Data Loading

```bash
python scripts/01_load_data.py
```

### 2. Processed Data Construction

```bash
python scripts/02_process_data.py
```

### 3. Model Fine-Tuning

```bash
python scripts/03_finetune_aurora.py
```

### 4. Final Evaluation

```bash
python scripts/04_test.py
```

### 5. Final Plot Generation

```bash
python src/plots/plot_result.py
```

## Generated Outputs

### Checkpoints

Training checkpoints are stored in:

```text
checkpoints/<phase>/lr_<learning_rate>/
```

Each directory may contain:

- `last.ckpt`
- the best checkpoint for that run;
- `completed.txt` when the configuration has already finished.

### Training Metrics

Per-epoch metrics and images are stored in:

```text
docs/entrenamiento/<phase>/lr_<learning_rate>/epoch_<xx>/
```

These usually include:

- `metricas.csv`
- `comparativa_viento.png`

### Final Results

Test results are stored in:

```text
docs/resultados/
```

These outputs include:

- CSV files with per-batch and per-step metrics;
- a summary plot comparing Aurora against persistence.

## Reference Baseline

Final evaluation compares Aurora against a persistence baseline:

- Aurora predicts future steps autoregressively.
- Persistence uses the last observed input state as the forecast.

The main project metrics are:

- Aurora RMSE
- Aurora MAE
- Persistence RMSE
- Skill score relative to persistence

## Project Status

This repository corresponds to an advanced stage of the Bachelor's Thesis, with
the pipeline already built and the training process almost completed. The
current code organization is intended to reflect the actual experimental
workflow used in the final project.

## Authorship

Bachelor's Thesis project focused on wind forecasting with ERA5, MongoDB, and
Aurora fine-tuning.
