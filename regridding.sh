#!/bin/bash
# Directories
INPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Targets_Rhires_TabsD"
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Inputs_regridded_RhiresD_TabsD"
GRID_FILE="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/RCM_grid.txt"

# Longitude & Latitude extent of Switzerland
LON_MIN=5.75553
LON_MAX=10.70298
LAT_MIN=45.64363
LAT_MAX=48.06181

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .nc files in the input directory
for file in "$INPUT_DIR"/*.nc; do
    filename=$(basename "$file")
    output_file="$OUTPUT_DIR/remapped_11km_${filename}"
    temp_regrid="${OUTPUT_DIR}/temp_regrid_${filename}"

    echo "Processing $filename ..."

    # Step 1: Regridding using bilinear interpolation with RCM grid
    cdo remapbil,"$GRID_FILE" "$file" "$temp_regrid"

    # Step 2: Cropping to Swiss lat/lon extent
    cdo sellonlatbox,$LON_MIN,$LON_MAX,$LAT_MIN,$LAT_MAX "$temp_regrid" "$output_file"

    # Remove temporary file
    rm "$temp_regrid"

    echo "Finished processing: $output_file"
done

echo "All files processed successfully!"
