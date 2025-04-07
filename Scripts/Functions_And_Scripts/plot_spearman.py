import numpy as np
import xarray as xr
from scipy.stats import spearmanr
import pyproj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_spearman(file1, var1_name, file2, var2_name, output_file):
    """This function calculates and plots Spearman's correlation between two variables after converting Swiss projection to WGS84."""
    
    # Load datasets
    ds1 = xr.open_mfdataset(file1, parallel=True)
    ds2 = xr.open_mfdataset(file2, parallel=True)
    
    # Extract the variables from the datasets
    var1 = ds1[var1_name]
    var2 = ds2[var2_name]
    
    # Ensure both variables have the same time dimension length (alignment)
    if not np.array_equal(var1.time, var2.time):
        raise ValueError("Time dimensions of the variables do not match.")
    
    # Define the Spearman correlation function
    def spearman(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 1:  # Ensure at least two valid data points for correlation
            return spearmanr(x[mask], y[mask])[0]
        else:
            return np.nan

    # Apply the Spearman correlation along the time dimension
    spearman_corr = xr.apply_ufunc(spearman, var1, var2,
                                   input_core_dims=[['time'], ['time']],
                                   vectorize=True,
                                   dask='allowed',  # Allows dask arrays if present
                                   output_dtypes=[np.float64])

    # Extract coordinate arrays for grid transformation
    lon_coords = ds1.lon.values  # Longitude coordinates
    lat_coords = ds1.lat.values  # Latitude coordinates

    # Set up the projection: Swiss projection (EPSG:21781) to WGS84 (EPSG:4326)
    swiss_proj = pyproj.CRS("EPSG:21781")
    wgs84_proj = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(swiss_proj, wgs84_proj, always_xy=True)
    
    # Convert grid coordinates from Swiss to WGS84
    lon, lat = transformer.transform(lon_coords, lat_coords)
    
    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Plot the Spearman correlation map
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Plot the correlation data
    corr_plot = ax.pcolormesh(lon_grid, lat_grid, spearman_corr, shading='auto', cmap='coolwarm')
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Add colorbar
    cbar = fig.colorbar(corr_plot, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Spearman Correlation")

    plt.title(f"Spearman Correlation between {var1_name} and {var2_name}")
    
    # Save the plot to the specified output file
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Example call to plot the correlation and save as PNG
    file1 = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/TabsD_1961_2023.nc"
    var1_name = "TabsD"
    file2 = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/RhiresD.nc"
    var2_name = "RhiresD"
    output_file = "spearman_correlation.png"  # Output file name

    plot_spearman(file1, var1_name, file2, var2_name, output_file)
