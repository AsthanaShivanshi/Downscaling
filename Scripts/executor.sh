#!/bin/bash

# Define the input files and variables
file1="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/TabsD_1961_2023.nc"
var1_name="TabsD"
file2="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/RhiresD.nc"
var2_name="RhiresD"
output_file="spearman_correlation.png"  # Output file name

# Run the Python script to generate the Spearman correlation plot
python3 plot_spearman.py "$file1" "$var1_name" "$file2" "$var2_name" "$output_file"
