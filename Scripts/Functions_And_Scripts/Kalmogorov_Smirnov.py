
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

    #Numpy arrays to store gridded values of KS stat and p values 

    ks_statistic_gridded = np.full ((lat,lon),np.nan)

    p_value_gridded = np.full((lat,lon), np.nan)

    #Regridding operation (might be required)

    if v1.shape!= v2.shape:
        v2= v2.interp_like (v1, method = "nearest")

    #Looping over grid cells 

    for i in range (lat):
        for j in range(lon):
            v1_ts= v1[:,i,j]
            v2_ts= v2[:,i,j]

            #In case of any NaNs , remove them
            mask= ~np.isnan(v1_ts) & ~np.isnan(v2_ts) 
            v1_ts= v1_ts [mask]
            v2_ts= v2_ts [mask]

            ks_statistic, p_value = stats.ks_2samp(v1_ts,v2_ts)
            ks_statistic_gridded[i,j]= ks_statistic
            p_value_gridded[i,j]= p_value


    #Plots for KS statistic and p values for two distributions
    plt.figure(figsize=(10, 6))
    plt.imshow(ks_statistic_gridded, cmap='viridis')
    plt.colorbar(label='Gridded KS Statistic for distributions {v1} and {v2}')
    plt.title(f'KS Statistic for each grid cell: {title or ""}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()

    # Plotting the p-value grid
    plt.figure(figsize=(10, 6))
    plt.imshow(p_value_gridded, cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(label='p-value gridded')
    plt.title(f'p-value for each grid cell: {title or ""}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()

    # Return the KS statistic and p-value grids for further analysis
    return ks_statistic_gridded, p_value_gridded


