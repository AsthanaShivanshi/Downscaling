#!/bin/bash
#SBATCH --job-name=regrid_cdo         # Job name
#SBATCH --output=regrid_cdo_%j.out    # Output file
#SBATCH --error=regrid_cdo_%j.err     # Error file
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --time=01:00:00               # Time limit
#SBATCH --mem=4GB                     # Memory allocation

# Load the necessary modules
module load python3 # Adjust based on the Python version you are using
module load cdo

# Run the Python script
python3 regridding.py

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Regridding completed successfully."
else
    echo "There was an error during the regridding process."
fi