#!/bin/bash
#SBATCH --job-name=KS_Tests      
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4     
#SBATCH --time=3-00:00:00         
#SBATCH --mem=32G  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python

source /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/activate

python correlation_Kendall_Spearman.py
