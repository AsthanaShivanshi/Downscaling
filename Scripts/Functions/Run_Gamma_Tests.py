from Gamma_KS_Test import Gamma_KS_gridded
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster
import sys

season_name=sys.argv[1] #Season is passed from the way it is written in Slurm script. Seasonwise processing, looping over seasons redundant
season_months_map = {
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5]
}
months=season_months_map.get(season_name)

ds2 = xr.open_dataset("/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/processed/Bicubic/Train/targets_precip_masked_train.nc", chunks={"time": 100})

RhiresD = ds2['RhiresD']

lon = RhiresD.lon
lat = RhiresD.lat
mask = np.isnan(lon) | np.isnan(lat)
RhiresD_gridded = RhiresD.where(~mask)

# Only wet days
RhiresD_wet = RhiresD_gridded.where(RhiresD_gridded >= 0.1)

mask_months=RhiresD_gridded["time"].dt.month.isin(months)

RhiresD_wet_season=RhiresD_wet.sel(time=mask_months)


#Running KS test for gamma gridpointwise for current season selected from the Slurm script

KS_Stat , p_val_ks_stat=Gamma_KS_gridded(RhiresD_wet_season,data_path=ds2,block_size=10, season=season_name)

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
