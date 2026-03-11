# Local CNN Training Pipeline

This folder converts the `machine_learning/microclimate_forecast_model.ipynb` notebook into a runnable Python package.

## Environment

TensorFlow 2.19 does not support Python 3.13, so this project is pinned to Python 3.12.

```powershell
cd machine_learning/local_cnn
poetry env use "C:\Program Files\Python312\python.exe"
poetry install
poetry run local-cnn-train --plot
```

## Outputs

Running the trainer writes artifacts into `artifacts/`:

- `best_pruned_model.keras`
- `pruned_float32.tflite`
- `pruned_int8.tflite`
- `feature_scaler.json`
- `training_summary.json`
- `weather_features.png` when `--plot` is used

## Useful options

```powershell
poetry run local-cnn-train --help
poetry run local-cnn-train --skip-open-meteo
poetry run local-cnn-train --open-meteo-start-date 2025-01-01 --open-meteo-end-date 2026-03-01
```
