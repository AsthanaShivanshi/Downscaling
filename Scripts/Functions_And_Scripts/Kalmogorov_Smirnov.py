
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def kalmogorov_smirnov(file1, var1, file2, var2, title=None): 
    """This function calculates gridded values of KS statistic as well as gridwise p values for gauging distance between distributions
    Inputs:
    file1,file2: Paths to Netcdf files
    var1,var2 : variable names such as TabsD and RhiresD on daily resolution
    title: Title of the plot, by default left as None
    """

    ds1= xr.open_dataset(file1)
    ds2= xr.open_dataset(file2)

    #Extracting the gridded variables
    v1= ds1[var1].values
    v2= ds2[var2].values

    # Numpy arrays to store gridded values of KS stat and p values 
    ks_statistic_gridded = np.full((v1.shape[1], v1.shape[2]), np.nan)
    p_value_gridded = np.full((v1.shape[1], v1.shape[2]), np.nan)

    # Regridding operation (might be required)
    if v1.shape != v2.shape:
        if 'lat' in v2.dims and 'lon' in v2.dims:
            v2 = v2.interp(lat=v1.lat, lon=v1.lon, method="nearest")
        elif 'y' in v2.dims and 'x' in v2.dims:
            v2 = v2.interp(y=v1.y, x=v1.x, method="nearest")
        else:
            raise ValueError("Unable to regrid: coordinate names not recognized")

    # Looping over grid cells
    for i in range(v1.shape[1]):
        for j in range(v1.shape[2]):
            v1_ts = v1[:, i, j]
            v2_ts = v2[:, i, j]

            # Removing NaNs in case there are any
            mask = ~np.isnan(v1_ts) & ~np.isnan(v2_ts) 
            v1_ts = v1_ts[mask]
            v2_ts = v2_ts[mask]

            ks_statistic, p_value = stats.ks_2samp(v1_ts, v2_ts)
            ks_statistic_gridded[i, j] = ks_statistic
            p_value_gridded[i, j] = p_value

    # Plots for KS statistic and p values
    plt.figure(figsize=(10, 6))
    plt.imshow(ks_statistic_gridded, cmap='viridis')
    plt.colorbar(label='Gridded KS Statistic for distributions {v1} and {v2}')
    plt.title(f'KS Statistic for each grid cell: {title or ""}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()

    # Plotting the gridded p-values
    plt.figure(figsize=(10, 6))
    plt.imshow(p_value_gridded, cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(label='p-value gridded')
    plt.title(f'p-value for each grid cell: {title or ""}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()

    # returning the gridded KS statistic and p-values
    return ks_statistic_gridded, p_value_gridded
