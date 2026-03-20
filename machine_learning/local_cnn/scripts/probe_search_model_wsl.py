import importlib.util

print("probe_start", flush=True)
spec = importlib.util.spec_from_file_location(
    "search_conservative_retrain",
    "/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn/scripts/search_conservative_retrain.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("module_loaded", flush=True)
print("visible_gpus_before", mod.tf.config.list_physical_devices("GPU"), flush=True)
model = mod.build_search_model(
    input_window_length=48,
    number_of_input_features=9,
    forecast_horizon_slots=24,
    temperature_feature_mean=25.0,
    temperature_feature_scale=2.0,
    temperature_feature_index=0,
)
print("model_built", flush=True)
print(model.name, flush=True)
