#!/bin/bash
#SBATCH --job-name=Gamma_tests     
#SBATCH --array=0-3 #For all four seasons , looping for generating plots
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=4         # 4 CPU cores (Dask + blocks)
#SBATCH --time=3-00:00:00        
#SBATCH --mem=64G                
#SBATCH --partition=cpu         
# (NO --gres=gpu:1 )
              

module load python

source /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/activate

#Directory containing the pipeline
cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/Scripts/Functions

SEASON_LIST=("JJA" "SON" "DJF" "MAM")

SEASON=${SEASON_LIST[$SLURM_ARRAY_TASK_ID]}

python Run_Gamma_Tests.py $SEASON

