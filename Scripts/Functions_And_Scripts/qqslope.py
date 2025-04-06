
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot_2samples
from scipy.stats import linregress

def qq_slope(file1, var1, file2, var2, title=None, cmap="coolwarm", quantiles=np.linspace(0.05, 0.95, 19), center_colorbar=False):
    """Computes and plots the gridded spatial map of QQ slopes based on probabilistic QQ plot
    Inputs: 
    file1,file2: NetCDF file paths
    var1,var2= variable names from the file1 and file2 resp. eg. TabsD and RhiresD
    title: optional plot title
    cmap: colorbar
    quantiles: probabilities 
    center_colorbar : False by default, can be changed to True if one needs colorbar for the gridded map of slopes centered around 0"""

    #Loading the datasets
    ds1= xr.open_dataset(file1)
    ds2= xr.open_dataset(file2)
    v1= ds1[var1]
    v2= ds2[var2]

    #Regridding might be required if the datasets are not of the same shapes
    if v1.shape != v2.shape:
        if 'lat' in v2.dims and 'lon' in v2.dims:
            v2 = v2.interp(lat=v1.lat, lon=v1.lon, method="nearest")
        elif 'y' in v2.dims and 'x' in v2.dims:
            v2 = v2.interp(y=v1.y, x=v1.x, method="nearest")
        else:
            raise ValueError("Unable to regrid: coordinate names not recognized")

    def qq_slope_from_probabilistic_quantiles(x, y):
        """Calculate QQ slope using probabilistic quantiles."""
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 10 or len(y) < 10:
            return np.nan  # not enough data

        try:
            qx = np.percentile(x, quantiles * 100) 
            qy = np.percentile(y, quantiles * 100)
            slope, _, _, _, _ = linregress(qx, qy)
            return slope
        except Exception:
            return np.nan

    # Applying function grid-wise to calculate the QQ slope
    slope_map = xr.apply_ufunc(
        qq_slope_from_probabilistic_quantiles, v1, v2,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[np.float64]
    )

    # Plot
    plt.figure(figsize=(10, 6))

    # Determine the color limits
    if center_colorbar:
        vmin = -np.nanmax(np.abs(slope_map.values))
        vmax = np.nanmax(np.abs(slope_map.values))
    else:
        vmin = slope_map.min().values
        vmax = slope_map.max().values

    slope_map.plot(cmap=cmap, add_colorbar=True, vmin=vmin, vmax=vmax)

    # Attempt to detect coordinate names (lat/lon or x/y)
    coords = slope_map.coords
    possible_x = [c for c in coords if 'lon' in c or c in ['x', 'easting', 'X']]
    possible_y = [c for c in coords if 'lat' in c or c in ['y', 'northing', 'Y']]
    x_coord = possible_x[0] if possible_x else list(coords)[-1]
    y_coord = possible_y[0] if possible_y else list(coords)[-2]

    plt.title(title or f"Grid-wise QQ Slope: {var1} vs {var2}")
    plt.xlabel(f"{x_coord} ({'km' if 'x' in x_coord.lower() else 'degrees'})")
    plt.ylabel(f"{y_coord} ({'km' if 'y' in y_coord.lower() else 'degrees'})")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return slope_map
