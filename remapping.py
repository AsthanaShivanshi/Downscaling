
#Regridding files to 12 kms resolution (not conservatively remapped for now)
import os
import subprocess

#Loading the data and specifying the directories
input_dir = '/Users/sasthana/Documents/Downscaling/data/Targets_RhiresD_TabsD'
output_dir = '/Users/sasthana/Documents/Downscaling/data/Inputs_regridded_RhiresD_TabsD'

#Defining the grid resolution file in .txt format
resolution_spec_file= '/Users/sasthana/Documents/data/grid_12_kms_file.txt'

for filename in os.listdir(input_dir):
    #Regridding loop after confirming files exist and are accessible
    if filename.endswith(".nc"):
        in_file = os.path.join(input_dir, filename)
        out_file= os.path.join(output_dir, f"remapped_{filename}")

    #Using cdo for remapping using bilinear interpolation
        cdo_command = [
                'cdo', 
                'remapbil,' + resolution_spec_file, 
                in_file, 
                out_file
            ]
        try:
            subprocess.run(cdo_command, check=True)
            print(f'regridding successfor {filename}')
            #Exception Handling
        except subprocess.CalledProcessError as e:
            print(f'Error regridding {filename}: {e}')
