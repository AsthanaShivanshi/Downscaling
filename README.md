# Downscaling
Downscaling of climate change projections using SR algorithms for climate change and extreme events
TabsD and RhiresD datasets (daily,averages) are in the data folder 
RCM dataset : 

How did I produce the regridded 12 km files from the 1 km dataset?
regridded using cdo remapbil to the grid resolution of the RCM 12 km file and saved in the regridded folder
For that, I used Historical —>0.11 degrees —→Daily mean —→2m air temperature and mean precipitation flux —>  MPI-M-MPI-ESM-LR—→ CLMcom-ETH-COSMO-crCLIM (Switzerland) —→ r1i1p1—> 2001-2005 downloaded from CDS

1. Limited the RCM files to 5E to 11E, 45N to 48N
2. Regridded RhiresD and TabsD using remapbil with the lat-lon file from 1