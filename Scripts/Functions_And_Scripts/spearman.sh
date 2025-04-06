#!/bin/bash

# Set the paths to your NetCDF files
file1="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/TabsD_1961_2023.nc"
file2="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/RhiresD_1961_2023.nc"
var1_name="TabsD"
var2_name="RhiresD"
output_file="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/spearman.nc"

# Run the Python script with the specified arguments
python3 /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/Scripts/Functions_And_Scripts/plot_spearman.py $file1 $var1_name $file2 $var2_name $output_file
