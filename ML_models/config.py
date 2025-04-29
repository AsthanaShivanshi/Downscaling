CONFIG = {
    "batch_size": 32,
    "num_workers": 4,
    "input_paths": {
        "precip_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/RhiresD_1km_bicubic_Swiss_features_masked.nc",
        "temp_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/TabsD_1km_bicubic_Swiss_features_masked.nc",
        "precip_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/raw/RhiresD_1971_2022.nc",
        "temp_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/raw/TabsD_1971_2022.nc",
    },
    "var_names": {
        "precip_input": "pr",
        "temp_input": "tas",
        "precip_target": "RhiresD",
        "temp_target": "TabsD",
    }
}
