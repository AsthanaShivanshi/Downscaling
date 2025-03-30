#!/bin/bash
# Directories
INPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Targets_Rhires_TabsD"
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Inputs_regridded_RhiresD_TabsD"

# Reference grid file
REFERENCE_GRID="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/tas_11km_limited.nc"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .nc files in the input directory
for file in "$INPUT_DIR"/*.nc; do
    filename=$(basename "$file")
    output_file="$OUTPUT_DIR/remapped_con_11km_${filename}"
    temp_regrid="${OUTPUT_DIR}/temp_regrid_${filename}"

    echo "Processing $filename ..."

    # Step 1: Regridding using bilinear interpolation to the grid of tas_11km_limited.nc
    cdo remapcon,"$REFERENCE_GRID" "$file" "$temp_regrid"

    # Step 2: Save the regridded output
    mv "$temp_regrid" "$output_file"

    echo "Finished processing: $output_file"
done

echo "All files processed successfully!"
