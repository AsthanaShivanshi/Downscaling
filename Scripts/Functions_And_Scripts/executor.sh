#!/bin/bash
#SBATCH --job-name=my_job       # Name of the job
#SBATCH --output=job_output-%j.txt # Standard output file
#SBATCH --error=job_error-%j.txt   # Standard error file
#SBATCH --ntasks=1              # Number of tasks (typically 1 for single-node jobs)
#SBATCH --cpus-per-task=4      # Number of CPUs per task (adjust as needed)
#SBATCH --time=02:00:00         # Time limit (format: HH:MM:SS)
#SBATCH --mem=64G                # Memory required (adjust as needed)

cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/Scripts/Functions_And_Scripts/

# Define the input files and variables
file1="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/TabsD_1961_2023.nc"
var1_name="TabsD"
file2="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/data_and_outputs/RhiresD.nc"
var2_name="RhiresD"
output_file="spearman_correlation.png"  # Output file name

# Run the Python script to generate the Spearman correlation plot
python3 plot_spearman.py "$file1" "$var1_name" "$file2" "$var2_name" "$output_file"
