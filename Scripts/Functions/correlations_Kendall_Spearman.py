import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from pyproj import Transformer
import os

def calculate_correlation(ds1_path, ds2_path):
    ds1 = xr.open_dataset(ds1_path)
    ds2 = xr.open_dataset(ds2_path)
    TabsD = ds1['TabsD']
    RhiresD = ds2['RhiresD']

    # Grid coordinates
    E = ds1["E"].values
    N = ds1["N"].values
    E_grid, N_grid = np.meshgrid(E, N)
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    # Define seasons
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11],
    }

    spearman_maps = {}
    kendall_maps = {}

    for season, months in seasons.items():
        TabsD_season = TabsD.sel(time=TabsD['time.month'].isin(months))
        RhiresD_season = RhiresD.sel(time=RhiresD['time.month'].isin(months))
        
        spearman_grid = np.full((len(N), len(E)), np.nan)
        kendall_grid = np.full((len(N), len(E)), np.nan)

        for i in range(len(N)):
            for j in range(len(E)):
                x = TabsD_season[:, i, j].values
                y = RhiresD_season[:, i, j].values

                mask = np.isfinite(x) & np.isfinite(y)
                if np.sum(mask) > 2:  # Need at least 3 points to correlate
                    spearman_corr, _ = spearmanr(x[mask], y[mask])
                    kendall_corr, _ = kendalltau(x[mask], y[mask])
                    spearman_grid[i, j] = spearman_corr
                    kendall_grid[i, j] = kendall_corr

        spearman_maps[season] = spearman_grid
        kendall_maps[season] = kendall_grid

    return lon, lat, spearman_maps, kendall_maps

def plot_correlations(lon, lat, correlation_maps, title_prefix, output_dir):
    fig_count = 0
    for season, corr in correlation_maps.items():
        plt.figure(figsize=(8,6))
        plt.pcolormesh(lon, lat, corr, shading='auto')
        plt.colorbar(label='Correlation')
        plt.title(f'{title_prefix} {season}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{title_prefix}_{season}.png")
        plt.savefig(filename)
        plt.close()
        fig_count += 1
    print(f"Saved {fig_count} plots for {title_prefix}")

def main():
    ds1_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/targets_tas_masked_train.nc"
    ds2_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/targets_precip_masked_train.nc"
    output_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/Outputs/plots"
    os.makedirs(output_dir, exist_ok=True)

    lon, lat, spearman_maps, kendall_maps = calculate_correlation(ds1_path, ds2_path)
    
    plot_correlations(lon, lat, spearman_maps, 'Spearman', output_dir)
    plot_correlations(lon, lat, kendall_maps, 'Kendall', output_dir)

if __name__ == "__main__":
    main()
