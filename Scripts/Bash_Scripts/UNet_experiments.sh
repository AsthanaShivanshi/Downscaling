#!/bin/bash
#SBATCH --job-name=UNet_training       # Name of the job
#SBATCH --output=job_output-%j.txt # Standard output file
#SBATCH --error=job_error-%j.txt   # Standard error file
#SBATCH --ntasks=1              # Number of tasks (typically 1 for single-node jobs)
#SBATCH --cpus-per-task=4      # Number of CPUs per task (adjust as needed)
#SBATCH --time=02:00:00         # Time limit (format: HH:MM:SS)
#SBATCH --mem=32G                # Memory required (adjust as needed)

#My environment activation
module purge
module load python

source /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/activate

#Containing the pipeline: the directory

cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models


#For quick test module
python Main.py --quick_test

#For full training modzle

#python Main.py 