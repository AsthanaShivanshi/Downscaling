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
