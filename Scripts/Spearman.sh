import xarray as xr
import numpy as np
from scipy.stats import spearmanr

# Define file paths
file_rhiresd = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Targets_Rhires_TabsD_Tmin_Tmax/concatenated_RhyresD_71_2000.nc"
file_tmax = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Targets_Rhires_TabsD_Tmin_Tmax/concatenated_Tmax_71_2000.nc"

# Load NetCDF files
ds_rhiresd = xr.open_dataset(file_rhiresd)
ds_tmax = xr.open_dataset(file_tmax)

var_rhiresd = ds_rhiresd['RhiresD']
var_tmax = ds_tmax['TmaxD']

# Check dimensions
if var_rhiresd.shape != var_tmax.shape:
    print("Regridding Tmax dataset to match RhiresD dimensions...")
    var_tmax = var_tmax.interp_like(var_rhiresd, method="linear")

# Spearman correlation function
def spearman_corr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)  # Ignore NaNs
    if mask.sum() > 1:
        return spearmanr(x[mask], y[mask])[0]
    else:
        return np.nan

# Apply Spearman correlation across grid points
spearman_corr_map = xr.apply_ufunc(
    spearman_corr, 
    var_rhiresd, var_tmax, 
    input_core_dims=[["time"], ["time"]], 
    vectorize=True,
    output_dtypes=[np.float64]
)

# Save results
output_file = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Ouputs_Plots_Files/spatial_spearman_correlation_precip_TmaxD.nc"
spearman_corr_map.to_netcdf(output_file)
