#!/usr/bin/env python3
import os
import subprocess

# Define input and output directories
input_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Targets_Rhires_TabsD"
output_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/Inputs_regridded_RhiresD_TabsD"

# Define grid file path
grid_file = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data/grid_12_kms_file.txt"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all .nc files in the input directory
for file in os.listdir(input_dir):
    if file.endswith(".nc"):
        input_file = os.path.join(input_dir, file)
        
        # Define the output file path with 'remapped_' prefix
        output_file = os.path.join(output_dir, f"remapped_{file}")

        # Print status message
        print(f"Regridding {input_file} to {output_file} using grid {grid_file}...")

        # Run CDO regridding command
        cdo_command = [
            'cdo', 
            f'remapbil,{grid_file}', 
            input_file, 
            output_file
        ]

        # Run the command and check if it was successful
        try:
            subprocess.run(cdo_command, check=True)
            print(f"Successfully regridded {file} to {output_file}.")
        except subprocess.CalledProcessError as e:
            print(f"Error regridding {file}: {e}")

print("All files regridded successfully!")
