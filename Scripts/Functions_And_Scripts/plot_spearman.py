import xarray as xr
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import kstest
import pandas as pd

def plot_spearman(file1, var1_name, file2, var2_name, title=None, cmap="coolwarm"):
    """This is a function written to plot Spearman correlation between two input files where:
    file1: input file1, e.g. TabsD for average daily temperature data
    file2: input file2, e.g. RhyresD for daily precipitation sum data
    var1_name: string with variable name in file1
    var2_name: string with variable name in file2
    title: title of the plot
    cmap: colormap for the plot
    """
    # Loading datasets
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    # Variables extraction from two files
    var1 = ds1[var1_name]
    var2 = ds2[var2_name]

    # Taking care of grid issues in case var1 and var2 don't lie on the same grid
    # Regridding var2 to the grid of var1 (might not be needed)
    if var1.shape != var2.shape:
        if 'lat' in var2.dims and 'lon' in var2.dims:
            # Regrid var2 to match var1's lat/lon grid using nearest interpolation
            var2 = var2.interp(lat=var1.lat, lon=var1.lon, method="nearest")
        elif 'y' in var2.dims and 'x' in var2.dims:
            # If coordinates are labeled as y/x, perform interpolation based on those
            var2 = var2.interp(y=var1.y, x=var1.x, method="nearest")
        else:
            # Raise an error if coordinate names are unrecognized
            raise ValueError(f"Unable to regrid: coordinate names not recognized. var2 coordinates are {var2.coords}")

    # Calculating Spearman's correlation
    def spearman(x, y):
        """This function returns Spearman correlation between two variables if they are not NaNs and returns NaN if either is NaN."""
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 1:
            return spearmanr(x[mask], y[mask])[0]
        else:
            return np.nan

    # Apply the spearman function to var1 and var2 along the time axis
    spearman_corr = xr.apply_ufunc(spearman, var1, var2,
                                   input_core_dims=[['time'], ['time']],
                                   vectorize=True,
                                   dask='allowed',
                                   output_dtypes=[np.float64])

    # Plotting the results obtained from the gridwise calculation
    plt.figure(figsize=(10, 6))
    spearman_corr.plot(cmap=cmap, add_colorbar=True)

    # Attempt to extract coordinate names (handle both lat/lon and x/y)
    coords = spearman_corr.coords
    possible_x = [c for c in coords if 'lon' in c or c in ['x', 'easting', 'X']]
    possible_y = [c for c in coords if 'lat' in c or c in ['y', 'northing', 'Y']]
    x_coord = possible_x[0] if possible_x else list(coords)[-1]
    y_coord = possible_y[0] if possible_y else list(coords)[-2]

    # Set plot title and labels
    plt.title(title if title else f"Spearman Correlation between {var1_name} and {var2_name}")
    plt.xlabel(f"{x_coord} ({'km' if 'x' in x_coord.lower() else 'degrees'})")
    plt.ylabel(f"{y_coord} ({'km' if 'y' in y_coord.lower() else 'degrees'})")

    # Optional: Convert ticks to km if projected coordinates (assuming x and y are in meters)
    if all(k in [x_coord.lower(), y_coord.lower()] for k in ['x', 'y']):
        ax = plt.gca()
        ax.set_xticklabels([f"{x/1000:.0f}" for x in ax.get_xticks()])
        ax.set_yticklabels([f"{y/1000:.0f}" for y in ax.get_yticks()])

    # Layout adjustments
    plt.tight_layout()
    plt.show()
