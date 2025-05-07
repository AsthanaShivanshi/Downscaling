from scipy.stats import kstest, norm, gamma
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import kstest
from pyproj import Transformer
import dask.array as da
from dask import delayed, compute
from das.diagnostics import ProgressBar

def Kalmogorov_Smirnov_Grid_Cell(tabsd_wet, mu, sigma, rhiresd_wet, alpha, beta, city_name="City"):
    """
    Prints KS statistics and p-values for temperature (assuming normal distribution) and precipitation 
    (assuming gamma distribution) for the grid cell specified considering their parameters are already available for that grid cell
    """
    # KS test (normal)
    ks_stat_tabsd, p_value_tabsd = kstest(tabsd_wet, "norm", args=(mu, sigma))
    print(f"KS statistic for average temperature on wet days in {city_name} is {ks_stat_tabsd:.3f} with a p-value of {p_value_tabsd:.3f}")

    # KS test (gamma)
    ks_stat_precip, p_value_precip = kstest(rhiresd_wet, "gamma", args=(alpha, 0, beta))  # gamma uses shape, loc, scale
    print(f"KS statistic for total daily precipitation on wet days in {city_name} is {ks_stat_precip:.3f} with a p-value of {p_value_precip:.3f}")


#xxxxxxxxxxxxxxxxxxxxxxKSTest for each grid cell in Siwtezrland###############

def Kalmogorov_Smirnov_gridded(temp, mean, std, data_path, alpha=0.05):
    """Performs KS test for each grid cell and plot the resulting gridwise plot along with rejedction/acceptance in accordance with 
    the p value (rejecting/accepting the null hypothesis)"""

    n_lat, n_lon = temp.sizes['N'], temp.sizes['E']
    KS_Stat = np.full((n_lat, n_lon), np.nan)
    p_val_ks_stat = np.full((n_lat, n_lon), np.nan)

    # Dask delayed list
    tasks = []

    for i in range(n_lat):
        for j in range(n_lon):
            data = temp[:, i, j].values
            mu = mean[i, j].values
            sigma = std[i, j].values

            task = delayed(ks_test_single_point)(data, mu, sigma)
            tasks.append((i, j, task))

    # Compute all tasks in parallel, with progress bar showing progress of the computation 
    with ProgressBar():

        results = compute(*[t[2] for t in tasks])

    # Assign results
    for idx, (i, j, _) in enumerate(tasks):
        stat, pval = results[idx]
        KS_Stat[i, j] = stat
        p_val_ks_stat[i, j] = pval

    E = data_path["E"].values
    N = data_path["N"].values
    E_grid, N_grid = np.meshgrid(E, N)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    # Binary mask: 1 if null accepted, 0 if rejected
    accept_h0 = (p_val_ks_stat > alpha).astype(int)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    cmap = plt.get_cmap('bwr', 2)  # blue=accepted, red=rejected

    plot = ax.pcolormesh(lon, lat, accept_h0, cmap=cmap, shading="auto", vmin=0, vmax=1, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(plot, ax=ax, shrink=0.7, orientation='horizontal', ticks=[0, 1])
    cbar.ax.set_xticklabels(['Reject H₀', 'Accept H₀'])
    cbar.set_label(f'KS Test Hypothesis Test (α={alpha})')
    plt.title('KS Test: Normality of Wet Day Temperature')
    plt.tight_layout()
    plt.show()

    return KS_Stat, p_val_ks_stat

# Helper function for dask
@delayed
def ks_test_single_point(data, mu, sigma):
    data = data[~np.isnan(data)]
    if len(data) > 0 and sigma > 0:
        stat, pval = kstest(data, 'norm', args=(mu, sigma))
        return stat, pval
    else:
        return np.nan, np.nan

