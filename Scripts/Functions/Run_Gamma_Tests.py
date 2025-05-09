from Gamma_KS_Test import Gamma_KS_gridded
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster
import sys

# Load dataset
ds2 = xr.open_dataset("/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/targets_precip_masked_train.nc", chunks={"time": 100})

RhiresD = ds2['RhiresD']

lon = RhiresD.lon
lat = RhiresD.lat
mask = np.isnan(lon) | np.isnan(lat)
RhiresD_gridded = RhiresD.where(~mask)

# Only wet days
RhiresD_wet = RhiresD_gridded.where(RhiresD_gridded >= 0.1)

# Define seasons
season_name=sys.argv[1]
months = {
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5]
}[season_name]

# Loop over seasons
for season_name, months in seasons.items():
    mask_months = RhiresD_wet['time'].dt.month.isin(months)
    RhiresD_wet_season = RhiresD_wet.sel(time=mask_months)

    KS_Stat, p_val_ks_stat = Gamma_KS_gridded(
        RhiresD_wet_season,
        data_path=ds2,
        block_size=20,
        season=season_name
    )

    print(f"Finished Gamma KS Test for {season_name}")


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
