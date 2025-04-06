
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot_2samples


def qqplot(file1, var1, file2,var2,title=None):
    """This function intends to generate the porbabilistic QQplot between two distributions, by first spatially averaging the data over the region of interest
    - file1, file2: paths to NetCDF files
    - var1, var2: eg TabsD and RhiresD for daily temp and precip for example
    - title: title for the plot, can be customised or left as None
    """

    #Loading the two datasets
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    #Extracting the variables from the two datasets
    var1 = ds1[var1]   
    var2 = ds2[var2]

    #Spatial averaging, over lat and lon, skipping NaNs
    var1_ts=var1.mean(dim=["lat","lon"], skipna=True).values
    var2_ts=var2.mean(dim=["lat","lon"], skipna=True).values

    #Drop NaNs if any
    mask = ~np.isnan(var1_ts) & ~np.isnan(var2_ts)
    var1_ts1 = var1_ts[mask]
    var2_ts2 = var2_ts[mask]

    #Plotting QQplot using the statsmodels library functionalities
    plt.figure(figsize=(10,6))
    qqplot_2samples(var1_ts,var2_ts, line="45") #Includes the 45 degrees line
    plt.xlabel(f"Quantiles of spatially averaged {var1}" )
    plt.ylabel(f"Quantiles of spatially averaged {var2}" )
    plt.title(title or f"QQ Plot : {var2} vs {var1}")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
