CONFIG = {
    "batch_size": 16,
    "num_workers": 4,
    "input_paths": {
        "precip_input": "path/to/RhiresD_1km_bicubic_Swiss_features_masked.nc",
        "temp_input": "path/to/TabsD_1km_bicubic_Swiss_features_masked.nc",
        "precip_target": "path/to/RhiresD_1971_2022.nc",
        "temp_target": "path/to/TabsD_1971_2022.nc",
    },
    "var_names": {
        "precip_input": "pr",
        "temp_input": "tas",
        "precip_target": "RhiresD",
        "temp_target": "TabsD",
    }
}
