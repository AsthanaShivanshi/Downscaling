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

#Directory containing the pipeline
cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/Scripts/Functions


#For quick test module uncomment
#python Main.py --quick_test

#For full training modzle utilisattion uncomment
#Model checkpoint will be saved at 20 epochs, then will continue from there to avoid losing everything

python Run_KS_Tests.py