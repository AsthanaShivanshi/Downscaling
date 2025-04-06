#!/bin/bash
#SBATCH --mem=16G  # Request 16 GB of memory

# Activate the specific Python environment
source /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/activate

# Execute the Python script with input arguments
/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/python /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/Scripts/Functions_And_Scripts/plot_spearman.py /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/TabsD_1961_2023.nc TabsD /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/RhiresD_1961_2023.nc RhiresD /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/spearman.nc
