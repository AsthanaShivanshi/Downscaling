import xarray as xr
import numpy as np

def gridded_R_squared(pred_path, truth_path, var1, var2, chunk_size={'time': 50}):
    """
    Calculate grid-wise coefficient of determination (R^2) between two gridded variables, ignoring NaNs.
    """
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    
    valid_mask = (~np.isnan(var1_data)) & (~np.isnan(var2_data))
    
    # Residual sum of squares
    ss_res = ((var1_data - var2_data) ** 2).where(valid_mask).sum(dim='time')
    
    # Total sum of squares
    mean_true = var2_data.where(valid_mask).mean(dim='time')
    ss_tot = ((var2_data - mean_true) ** 2).where(valid_mask).sum(dim='time')
    
    r2 = 1 - (ss_res / ss_tot)
    r2 = r2.astype(np.float32)
    
    return r2

def pooled_R_squared(pred_path, truth_path, var1, var2, chunk_size={'time': 50}):
    """
    Calculate a single pooled coefficient of determination (R^2) between two gridded variables,
    pooling across all spatial and temporal dimensions, ignoring NaNs.
    """
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    
    var1_flat = var1_data.stack(points=('time', 'lat', 'lon'))
    var2_flat = var2_data.stack(points=('time', 'lat', 'lon'))
    
    valid_mask = (~np.isnan(var1_flat)) & (~np.isnan(var2_flat))
    
    var1_valid = var1_flat.where(valid_mask, drop=True)
    var2_valid = var2_flat.where(valid_mask, drop=True)
    
    ss_res = ((var1_valid - var2_valid) ** 2).sum().item()
    
    mean_true = var2_valid.mean()
    ss_tot = ((var2_valid - mean_true) ** 2).sum().item()
    
    r2 = 1 - (ss_res / ss_tot)
    
    return np.float32(r2)

