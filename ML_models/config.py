CONFIG = {
    "batch_size": 32,
    "num_workers": 4,
    "input_paths": {
        "precip_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/SCALED_features_precip_masked_bicubic_train.nc",
        "temp_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/SCALED_features_tas_masked_bicubic_train.nc",
        "precip_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/SCALED_targets_precip_masked_train.nc",
        "temp_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/SCALED_targets_tas_masked_train.nc",
    },
    "var_names": {
        "precip_input": "pr",
        "temp_input": "tas",
        "precip_target": "RhiresD",
        "temp_target": "TabsD",
    }
}
