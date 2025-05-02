CONFIG = {
    "batch_size": 32,
    "num_workers": 4,
    "input_paths": {
        # Train
        "precip_train_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/SCALED_features_precip_masked_bicubic_train.nc",
        "temp_train_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/features_tas_masked_bicubic_train.nc",
        "precip_train_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/SCALED_targets_precip_masked_train.nc",
        "temp_train_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/SCALED_targets_tas_masked_train.nc",
        
        # Validation
        "precip_val_input": "/path/to/precip_val_input.nc",
        "temp_val_input": "/path/to/temp_val_input.nc",
        "precip_val_target": "/path/to/precip_val_target.nc",
        "temp_val_target": "/path/to/temp_val_target.nc",
        
        # Test
        "precip_test_input": "/path/to/precip_test_input.nc",
        "temp_test_input": "/path/to/temp_test_input.nc",
        "precip_test_target": "/path/to/precip_test_target.nc",
        "temp_test_target": "/path/to/temp_test_target.nc",
    },
    "var_names": {
        "precip_input": "pr",
        "temp_input": "tas",
        "precip_target": "RhiresD",
        "temp_target": "TabsD",
    }
}
