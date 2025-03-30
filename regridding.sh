#!/bin/bash

# Directories
INPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Targets_Rhires_TabsD"
OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Inputs_regridded_RhiresD_TabsD"
GRID_FILE="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/tas_11km_Switzerland.nc"

# Lon and Lat extents of the final output files
LON_MIN=5.75553
LON_MAX=10.70298
LAT_MIN=45.64363
LAT_MAX=48.06181


# Looping through all .nc files in the Targets_RhiresD_tabsD directory

for file in "$INPUT_DIR"/*.nc; do
    # Extracting the filename
    filename=$(basename "$file")

    # Defining the output filename
    output_file="$OUTPUT_DIR/remapped_11km_${filename}"

    echo "Processing $filename ..."

    # Step 1: Regridding using bilinear interp.
    temp_regrid="${OUTPUT_DIR}/temp_regrid_${filename}"
    cdo remapbil,"$GRID_FILE" "$file" "$temp_regrid"

    # Step 2: Cropping to the required lat/lon extent
    cdo sellonlatbox,$LON_MIN,$LON_MAX,$LAT_MIN,$LAT_MAX "$temp_regrid" "$output_file"

    # Removing temporary file
    rm "$temp_regrid"

    echo "Finished processing: $output_file"
done

echo "All files processed successfully!"
