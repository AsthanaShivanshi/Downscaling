#!/bin/bash
# Directories
INPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Targets_Rhires_TabsD_Tmin_Tmax"
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Inputs_regridded_RhiresD_TabsD_Tmin_Tmax"

# Reference grid file
REFERENCE_GRID="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/tas_11km_limited.nc"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Input concatenated file
input_file="${INPUT_DIR}/concatenated_Tmax_71_2000.nc"
output_file="${OUTPUT_DIR}/regridded_11km_concatenated_Tmax_71_2000.nc"
temp_regrid="${OUTPUT_DIR}/temp_regrid_concatenated_Tmax_71_2000.nc"

echo "Processing concatenated file..."

# Step 1: Regridding using bilinear interpolation to the grid of tas_11km_limited.nc
cdo remapbil,"$REFERENCE_GRID" "$input_file" "$temp_regrid"

# Step 2: Save the regridded output
mv "$temp_regrid" "$output_file"

echo "Finished processing: $output_file"
echo "File regridded successfully!"
