#!/bin/bash
#SBATCH --job-name=UNet_training_ALL_samples_Cyclical_LR     
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4     
#SBATCH --time=3-00:00:00         
#SBATCH --mem=64G  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
              

module load python

source /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/activate

#Directory containing the pipeline
cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models/Cyclical_LR_UNet

export WANDB_MODE="online"

#For quick test module uncomment
python Main.py --quick_test

#For full training modzle utilisattion uncomment

#python Main.py