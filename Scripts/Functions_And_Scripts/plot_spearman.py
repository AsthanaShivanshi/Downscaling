import numpy as np
import xarray as xr
import pyproj
from scipy.stats import spearmanr
import sys 

def project_curvilinear_to_latlon(E, N, projection_str="epsg:3395"):
    """Project curvilinear E, N coordinates to lat, lon using pyproj."""
    # Flatten the input arrays
    E_flat = E.values.flatten() if isinstance(E, xr.DataArray) else E.flatten()
    N_flat = N.values.flatten() if isinstance(N, xr.DataArray) else N.flatten()

    # Define the transformer from projection
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(3395),  # Source projection (Mercator)
        pyproj.CRS.from_epsg(4326)   # Destination projection (WGS84 lat-lon)
    )

    # Transform the coordinates
    lon_flat, lat_flat = transformer.transform(E_flat, N_flat)

    # Reshape back to original grid shape
    lon = lon_flat.reshape(E.shape)
    lat = lat_flat.reshape(N.shape)

    return lon, lat

def regrid_to_regular_grid(var1, var2, target_lat, target_lon):
    """
    This function interpolates the curvilinear grid data (var1, var2) to a regular lat-lon grid.
    """
    # Regrid the variables to the target lat/lon grid using nearest interpolation
    var1_interp = var1.interp(lat=target_lat, lon=target_lon, method="nearest")
    var2_interp = var2.interp(lat=target_lat, lon=target_lon, method="nearest")
    
    return var1_interp, var2_interp

def resample_dataset(ds1, ds2):
    """
    Resample ds2 to match the 'N' dimension of ds1.
    """
    target_N = ds1["N"]
    ds2_resampled = ds2.interp(N=target_N)
    return ds2_resampled

def plot_spearman(file1, var1_name, file2, var2_name, output_file="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/spearman.nc"):
   
    """This function calculates and saves Spearman's correlation between two variables."""
    
    # Load datasets
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)
    
    # Resample ds2's N to match ds1's N (i.e., make the latitudes compatible)
    if ds1["N"].shape != ds2["N"].shape:
        print(f"Resampling ds2 to match ds1's N dimension...")
        ds2_resampled = resample_dataset(ds1, ds2)
        var2_resampled = ds2_resampled[var2_name]
        print(f"Resampled ds2 N: {ds2_resampled['N'].shape}")
    else:
        var2_resampled = ds2[var2_name]
        print("ds2's N dimension matches ds1's N dimension. No resampling required.")
    
    # Extract variables from datasets
    var1 = ds1[var1_name]
    
    # Get E and N coordinates (the curvilinear grid)
    E1, N1 = ds1["E"], ds1["N"]
    E2, N2 = ds2["E"], ds2["N"]
    
    # Project the curvilinear grids to lat-lon
    lon1, lat1 = project_curvilinear_to_latlon(E1, N1)
    lon2, lat2 = project_curvilinear_to_latlon(E2, N2)
    
    # Create a common lat-lon grid (e.g., the target grid could be from var1 or var2)
    target_lat = np.linspace(np.min(lat1), np.max(lat1), len(lat1))
    target_lon = np.linspace(np.min(lon1), np.max(lon1), len(lon1))
    
    # Regrid the variables to the regular lat-lon grid
    var1_interp, var2_interp = regrid_to_regular_grid(var1, var2_resampled, target_lat, target_lon)
    
    # Calculate Spearman's correlation
    def spearman(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 1:
            return spearmanr(x[mask], y[mask])[0]
        else:
            return np.nan

    spearman_corr = xr.apply_ufunc(spearman, var1_interp, var2_interp,
                                   input_core_dims=[['time'], ['time']],
                                   vectorize=True,
                                   dask='allowed',
                                   output_dtypes=[np.float64])
    
    # Save the Spearman correlation to a NetCDF file
    spearman_corr.to_netcdf(output_file)
    print(f"Spearman correlation saved to {output_file}")

if __name__ == "__main__":
    # Get command-line arguments
    file1 = sys.argv[1]
    var1_name = sys.argv[2]
    file2 = sys.argv[3]
    var2_name = sys.argv[4]
    output_file = sys.argv[5] if len(sys.argv) > 5 else "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/spearman.nc"
    
    # Call the plot_spearman function
    plot_spearman(file1, var1_name, file2, var2_name, output_file)
