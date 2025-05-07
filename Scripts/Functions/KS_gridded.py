from scipy.stats import kstest
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
from dask import delayed, compute
from dask.diagnostics import ProgressBar

ProgressBar().register()

# Helper function for Dask (for processing a block)
@delayed
def process_block(temp, mean, std, i_start, i_end, j_start, j_end):
    block_KS = np.full((i_end - i_start, j_end - j_start), np.nan)
    block_pval = np.full((i_end - i_start, j_end - j_start), np.nan)

    for ii in range(i_start, i_end):
        for jj in range(j_start, j_end):
            data = temp[:, ii, jj]
            mu = mean[ii, jj]
            sigma = std[ii, jj]
            data = data[~np.isnan(data)]
            if len(data) > 0 and sigma > 0:
                stat, pval = kstest(data, 'norm', args=(mu, sigma))
                block_KS[ii - i_start, jj - j_start] = stat
                block_pval[ii - i_start, jj - j_start] = pval

    return block_KS, block_pval

# Main function
def Kalmogorov_Smirnov_gridded(temp, mean, std, data_path, alpha=0.05, block_size=20, season="Season"):
    """Performs KS test for each grid cell and plot the resulting gridwise plot along with rejection/acceptance in accordance with 
    the p value (rejecting/accepting the null hypothesis)"""

    n_lat, n_lon = temp.sizes['N'], temp.sizes['E']
    KS_Stat = np.full((n_lat, n_lon), np.nan)
    p_val_ks_stat = np.full((n_lat, n_lon), np.nan)

    # Dask delayed list
    tasks = []

    for i in range(0, n_lat, block_size):
        for j in range(0, n_lon, block_size):
            i_end = min(i + block_size, n_lat)
            j_end = min(j + block_size, n_lon)
            task = process_block(temp.values, mean.values, std.values, i, i_end, j, j_end)
            tasks.append((i, j, task))

    # Compute all tasks in parallel, with progress bar showing progress of the computation 
    with ProgressBar():
        results = compute(*[t[2] for t in tasks], scheduler="threads")

    # Assign results back to full grid
    for idx, (i, j, _) in enumerate(tasks):
        block_KS, block_pval = results[idx]
        KS_Stat[i:i+block_KS.shape[0], j:j+block_KS.shape[1]] = block_KS
        p_val_ks_stat[i:i+block_pval.shape[0], j:j+block_pval.shape[1]] = block_pval

    # Prepare grid for plotting
    E = data_path["E"].values
    N = data_path["N"].values
    E_grid, N_grid = np.meshgrid(E, N)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    # Binary mask: 1 if null accepted, 0 if rejected
    accept_h0 = (p_val_ks_stat > alpha).astype(int)

    # Plotting
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
    plt.title(f'KS Test: Normality of Wet Day Temperature - {season}')
    plt.tight_layout()

    # Save plot to file
    plt.savefig(f"KS_Test_{season}.png", dpi=300)
    plt.close()

    return KS_Stat, p_val_ks_stat
