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
        "precip_val_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Val/SCALED_features_precip_masked_bicubic_val.nc",
        "temp_val_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Val/SCALED_features_tas_masked_bicubic_val.nc",
        "precip_val_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Val/SCALED_targets_precip_masked_val.nc",
        "temp_val_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Val/SCALED_targets_tas_masked_val.nc",
        
        # Test
        "precip_test_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Test/SCALED_features_precip_masked_bicubic_test.nc",
        "temp_test_input": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Test/SCALED_features_tas_masked_bicubic_test.nc",
        "precip_test_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Test/SCALED_targets_precip_masked_test.nc",
        "temp_test_target": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Test/SCALED_targets_tas_masked_test.nc",
    },
    "var_names": {
        "precip_input": "pr",
        "temp_input": "tas",
        "precip_target": "RhiresD",
        "temp_target": "TabsD",
    }
}
