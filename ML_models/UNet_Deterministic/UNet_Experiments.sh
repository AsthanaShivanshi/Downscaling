#!/bin/bash
#SBATCH --job-name=UNet_training_100_samples_Triangular2_LR    
#SBATCH --output=/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/UNet_Deterministic/job_output-Cyclical_LR-%j.txt 
#SBATCH --error=/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/UNet_Deterministic/job_error-Cyclical_LR%j.txt  
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4     
#SBATCH --time=3-00:00:00         
#SBATCH --mem=64G  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
              

module load python

source /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/activate

#Directory containing the pipeline
cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/UNet_Deterministic

export WANDB_MODE="online"

#For quick test module uncomment
python Main.py --quick_test

#For full training modzle utilisattion uncomment

#python Main.py